"""Native Clover block diffusion on the InfiniLM C++ model and InfiniOps."""

import time
from dataclasses import dataclass

from .frontier import FrontierPolicy


@dataclass
class CloverResult:
    text: str
    token_ids: list[int]
    iterations: list[int]
    elapsed_s: float

    @property
    def output_tokens_per_second(self):
        return len(self.token_ids) / self.elapsed_s


class CloverGenerator:
    """Run greedy diffusion with clean prefix state and causal block replay."""

    def __init__(self, model_path, *, tp=2, max_context=4096):
        import torch
        from transformers import AutoTokenizer

        from infinilm import _core as core
        from infinilm.cache import StaticKVCacheConfig
        from infinilm.distributed import DistConfig
        from infinilm.infer_engine import InferEngine
        from infinilm.modeling_utils import load_model_state_dict_by_file

        self.torch = torch
        self.core = core
        self.cache_config = StaticKVCacheConfig(1, max_context)
        self.engine = InferEngine(
            model_path,
            device=core.device("hygon"),
            distributed_config=DistConfig(tp),
            cache_config=self.cache_config,
        )
        if self.engine.model_type != "clover_moe":
            raise ValueError("CloverGenerator requires a Clover checkpoint.")
        load_model_state_dict_by_file(self.engine, model_path, self.engine.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.mask_id = self.engine.hf_config["mask_token_id"]
        self.block_size = self.engine.hf_config["block_size"]
        self.max_context = max_context

    def _tensor(self, values, *, wide=False):
        import numpy as np

        return self.core.from_numpy(
            np.asarray(values, dtype=np.int64 if wide else np.int32)
        )

    def _forward(self, ids, past, *, preview=False, expert_budget=False):
        length = len(ids)
        return self.engine.forward_raw(
            self._tensor([ids], wide=True),
            position_ids=self._tensor([list(range(past, past + length))], wide=True),
            past_kv_lengths=self._tensor([past]),
            total_kv_lengths=self._tensor([past + length]),
            input_offsets=self._tensor([0, length]),
            cu_seqlens=self._tensor([0, past + length]),
            block_tables=self._tensor([[0]]),
            mamba_init_state_indices=self._tensor([0]),
            mamba_final_state_indices=self._tensor([0]),
            diffusion_preview=preview,
            diffusion_expert_budget=expert_budget,
            sample_all_positions=preview,
            temperature=0,
            top_k=1,
        )

    def _logits(self, value):
        torch, core = self.torch, self.core
        out = torch.empty(value.shape, device="cuda:0", dtype=torch.bfloat16)
        target = core.from_blob(
            out.data_ptr(),
            list(value.shape),
            dtype=core.bfloat16,
            device=core.device("hygon", 0),
            owner=out,
        )
        target.copy_(value)
        core.set_device(core.device("hygon", 0))
        core.sync_device()
        return out.view(self.block_size, -1).float()

    def generate(
        self,
        prompt,
        *,
        max_new_tokens=128,
        policy="baseline",
        threshold=0.95,
        ignore_eos=False,
        expert_budget=False,
        frontier_confidence_floor=0.9,
    ):
        if policy not in ("baseline", "frontier"):
            raise ValueError("Policy must be baseline or frontier.")
        if max_new_tokens < 0 or not 0 <= threshold <= 1:
            raise ValueError("Invalid output limit or confidence threshold.")
        torch = self.torch
        prompt_ids = self.tokenizer.apply_chat_template(
            (
                [{"role": "user", "content": prompt}]
                if isinstance(prompt, str)
                else prompt
            ),
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if hasattr(prompt_ids, "keys"):
            prompt_ids = prompt_ids["input_ids"]
        blocks = (max_new_tokens + self.block_size - 1) // self.block_size
        if len(prompt_ids) + blocks * self.block_size > self.max_context:
            raise ValueError(
                "Prompt and complete diffusion blocks exceed the cache capacity."
            )
        self.engine.reset_cache(self.cache_config)
        start = time.perf_counter()
        if max_new_tokens == 0:
            return CloverResult("", [], [], time.perf_counter() - start)
        self._forward(prompt_ids, 0)
        output, iterations = [], []
        frontier = FrontierPolicy(confidence_floor=frontier_confidence_floor)
        eos = set(self.engine.eos_token_id)
        for _ in range(blocks):
            ids = torch.full(
                (self.block_size,), self.mask_id, device="cuda:0", dtype=torch.int64
            )
            for iteration in range(self.block_size):
                result = self._forward(
                    ids.cpu().tolist(),
                    len(prompt_ids) + len(output),
                    preview=True,
                    expert_budget=expert_budget,
                )
                logits = self._logits(result["logits"])
                logits[:, self.mask_id] = -float("inf")
                confidence, proposed = logits.softmax(-1).max(-1)
                masked = ids == self.mask_id
                if policy == "frontier":
                    selected = frontier.select(
                        confidence[None], masked[None], iteration
                    )[0]
                else:
                    selected = masked & (confidence >= threshold)
                    if not bool(selected.any()):
                        selected[confidence.masked_fill(~masked, -1).argmax()] = True
                ids = torch.where(selected, proposed, ids)
                if not bool((ids == self.mask_id).any()):
                    break
            resolved = ids.cpu().tolist()
            if self.mask_id in resolved:
                raise RuntimeError("An unresolved diffusion block cannot be committed.")
            self._forward(resolved, len(prompt_ids) + len(output))
            iterations.append(iteration + 1)
            remaining = max_new_tokens - len(output)
            for token in resolved[:remaining]:
                if not ignore_eos and token in eos:
                    return CloverResult(
                        self.tokenizer.decode(output, skip_special_tokens=True),
                        output,
                        iterations,
                        time.perf_counter() - start,
                    )
                output.append(token)
        return CloverResult(
            self.tokenizer.decode(output, skip_special_tokens=True),
            output,
            iterations,
            time.perf_counter() - start,
        )
