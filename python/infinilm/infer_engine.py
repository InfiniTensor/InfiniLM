import logging
import time
from dataclasses import dataclass
from typing import List, Optional

import infinicore
import torch

logger = logging.getLogger(__name__)

from infinilm.cache import StaticKVCacheConfig, PagedKVCacheConfig
from infinilm.distributed import DistConfig
from infinilm.lib import _infinilm

from .modeling_utils import parse_dtype
from .exception_utils import handle_oom_and_exit
import json
import os


def read_hf_config(model_path):
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    if "model_type" not in config_dict:
        raise ValueError(
            f"`model_type` is not specified in the config file `{config_path}`."
        )
    return config_dict

# config.json (required) defines model architecture, while generation_config.json
# (optional) defines generation behavior. They are kept as separate readers
# because: 1) config.json must exist and requires model_type validation,
# whereas generation_config.json may not exist; 2) keeping them separate
# preserves clear semantics and avoids a one-size-fits-all function with
# multiple conditional parameters.
def read_hf_generation_config(model_path):
    gen_config_path = os.path.join(model_path, "generation_config.json")
    if os.path.exists(gen_config_path):
        with open(gen_config_path, "r") as f:
            return json.load(f)
    return {}

@dataclass
class GenerationConfig:
    max_new_tokens: int | None = None

    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0

    eos_token_id: list[int] | None = None
    stop_on_eos: bool = True


class InferEngine(_infinilm.InferEngine):
    def __init__(
        self,
        model_path,
        device=None,
        distributed_config=DistConfig(1),
        cache_config=None,
        enable_graph_compiling=False,
        attention_backend="default",
        kv_cache_dtype=None,
    ):
        self.hf_config = read_hf_config(model_path)
        self.hf_generation_config = read_hf_generation_config(model_path)
        self._model_path = model_path

        if device is None:
            device = infinicore.device()
        self._infini_device = device

        hf_config_str = json.dumps(self.hf_config)
        super().__init__(
            hf_config_str,
            distributed_config._underlying,
            device._underlying.type,
            cache_config,
            enable_graph_compiling,
            attention_backend,
            (
                parse_dtype(kv_cache_dtype)._underlying
                if kv_cache_dtype is not None
                else None
            ),
        )
        self.use_cache = False

        self.enable_paged_attn = isinstance(cache_config, PagedKVCacheConfig)

        self._compiled_prefill_runner = None
        self._compiled_prefill_ready = False
        self._paged_kv_layers_cache = None
        try:
            from infinilm.compile.env import prefill_compile_enabled

            self._prefill_compile_enabled = prefill_compile_enabled()
        except ImportError:
            self._prefill_compile_enabled = False

        self._maybe_bootstrap_compiled_subgraphs()

    def _maybe_bootstrap_compiled_subgraphs(self):
        try:
            from infinicore.compiled_subgraphs import (
                any_compiled_subgraph_flag_enabled,
                bootstrap_from_infinicore_device,
            )
        except ImportError:
            return
        if not any_compiled_subgraph_flag_enabled():
            return
        from infinicore.utils import to_torch_dtype

        hidden = int(self.hf_config.get("hidden_size", 3584))
        bootstrap_from_infinicore_device(
            infini_device=self._infini_device,
            hidden_size=hidden,
            dtype=to_torch_dtype(self.dtype),
            warmup=True,
        )

    def _hybrid_prefill_ready(self) -> bool:
        """True when hybrid may run (CUDAGraph capture finished if enabled)."""
        runner = self._compiled_prefill_runner
        if runner is None:
            return False
        from infinilm.compile.env import prefill_cudagraph_enabled

        if prefill_cudagraph_enabled() and not runner._cudagraph_capture_done:
            return False
        return True

    def _compiled_prefill_supported(self) -> bool:
        if not self._prefill_compile_enabled:
            return False
        if not self.enable_paged_attn:
            return False
        if self.hf_config.get("model_type") not in ("fm9g", "fm9g7b", "llama", "minicpm"):
            return False
        return True

    def _get_paged_kv_layers(self):
        """Cached paged KV layer handles for share-weights hybrid prefill."""
        if self._paged_kv_layers_cache is not None:
            return self._paged_kv_layers_cache
        self._paged_kv_layers_cache = super().get_paged_kv_cache_tensors()
        return self._paged_kv_layers_cache

    def _ensure_compiled_prefill_runner(self, *, block_size: int = 256):
        if not self._compiled_prefill_supported():
            return
        if self._compiled_prefill_runner is not None:
            return

        from infinilm.compile import CompiledPrefillConfig, CompiledPrefillRunner
        from infinilm.compile.env import (
            compile_bucket_mode,
            compile_bucket_step,
            compile_max_seq_len,
            compile_warmup_seq_lens,
            prefill_cudagraph_enabled,
            prefill_share_weights_enabled,
        )
        from infinicore.utils import to_torch_dtype

        max_seq = compile_max_seq_len()
        cfg = CompiledPrefillConfig(
            model_path=self._model_path,
            max_seq_len=max_seq,
        )
        torch_device = torch.device("cuda", 0)
        cpp_state_dict = None
        kv_layers = None
        if prefill_share_weights_enabled():
            cpp_state_dict = super().state_dict()[0]
            if self._paged_kv_layers_cache is not None:
                kv_layers = self._paged_kv_layers_cache
            if self.enable_paged_attn:
                cache_cfg = self.get_cache_config()
                if cache_cfg is not None:
                    block_size = cache_cfg.block_size()
        warmup_seq_lens = compile_warmup_seq_lens(max_seq)
        logger.info(
            "compiled prefill warmup seq_lens=%s mode=%s step=%s",
            warmup_seq_lens,
            compile_bucket_mode(),
            compile_bucket_step(),
        )
        runner = CompiledPrefillRunner(
            cfg,
            device=torch_device,
            dtype=to_torch_dtype(self.dtype),
            warmup_seq_lens=warmup_seq_lens,
            cpp_state_dict=cpp_state_dict,
            kv_layers=kv_layers,
            block_size=block_size,
        )
        self._compiled_prefill_runner = runner
        self._compiled_prefill_ready = True
        if (
            kv_layers is not None
            and prefill_share_weights_enabled()
            and prefill_cudagraph_enabled()
        ):
            runner.ensure_cudagraph_capture(kv_layers, block_size=block_size)
        from infinilm.compile.mem_profile import snapshot_gpu_mem

        snapshot_gpu_mem("T3_server_idle")

    @staticmethod
    def _sample_token_id_from_logits(
        logits_1d: torch.Tensor,
        generation_config: GenerationConfig,
    ) -> int:
        """Match C++ ``random_sample`` on 1-D last-token logits; return scalar token id."""
        import random

        from infinicore.nn.functional import random_sample
        from infinilm.generation.utils import infini_to_numpy

        logits_infini = infinicore.from_torch(logits_1d.reshape(-1).contiguous())
        out = random_sample(
            logits_infini,
            random.random(),
            generation_config.top_p,
            generation_config.top_k,
            generation_config.temperature,
        )
        return int(infini_to_numpy(out).reshape(-1)[0])

    @staticmethod
    def _infinicore_input_ids_to_torch(input_ids) -> torch.Tensor:
        if not isinstance(input_ids, infinicore.Tensor):
            input_ids = infinicore.Tensor(input_ids)
        if input_ids.device.type != "cuda":
            input_ids = input_ids.to(infinicore.device("cuda", 0))
        return infinicore.to_torch(input_ids.contiguous())

    def try_hybrid_prefill_forward(self, **model_input) -> Optional[List[int]]:
        """Server/scheduler prefill: hybrid torch compile when eligible (batch size 1)."""
        if not self._compiled_prefill_supported() or not self.enable_paged_attn:
            return None
        past_kv_lengths = model_input.get("past_kv_lengths")
        if past_kv_lengths is None or past_kv_lengths.shape[0] != 1:
            return None
        block_tables = model_input.get("block_tables")
        if block_tables is None:
            return None

        input_ids = model_input["input_ids"]
        seq_len = int(input_ids.shape[-1])
        from infinilm.compile.env import (
            compile_buckets,
            compile_max_seq_len,
            prefill_cudagraph_enabled,
        )
        from infinilm.compile.runner import min_compiled_prefill_seq_len

        if not self._hybrid_prefill_ready():
            return None

        min_seq = min_compiled_prefill_seq_len()
        if prefill_cudagraph_enabled():
            # C-Eval shorts: stay on C++ until smallest compile bucket (512).
            min_seq = max(min_seq, min(compile_buckets(compile_max_seq_len())))
        if seq_len < min_seq:
            return None

        paged_block_size = self.get_cache_config().block_size()
        cpp_prefill_kwargs = {
            k: model_input[k]
            for k in (
                "input_ids",
                "position_ids",
                "past_kv_lengths",
                "total_kv_lengths",
                "input_offsets",
                "cu_seqlens",
                "block_tables",
                "slot_mapping",
                "temperature",
                "top_k",
                "top_p",
            )
            if k in model_input
        }
        from infinilm.compile.env import prefill_share_weights_enabled

        logger.debug(
            "compiled prefill (server path, share_weights=%s), seq_len=%s",
            prefill_share_weights_enabled(),
            model_input["input_ids"].shape[-1],
        )
        token_id = self._hybrid_compiled_prefill_step(
            model_input["input_ids"],
            input_ids_torch=model_input.get("input_ids_torch"),
            temperature=float(model_input.get("temperature", 1.0)),
            top_k=int(model_input.get("top_k", 1)),
            top_p=float(model_input.get("top_p", 1.0)),
            block_tables=block_tables,
            slot_mapping=model_input.get("slot_mapping"),
            paged_block_size=paged_block_size,
            cpp_prefill_kwargs=cpp_prefill_kwargs,
        )
        return [token_id]

    def _hybrid_compiled_prefill_step(
        self,
        input_ids,
        *,
        input_ids_torch: Optional[torch.Tensor] = None,
        temperature: float,
        top_k: int,
        top_p: float,
        block_tables,
        slot_mapping,
        paged_block_size: int,
        cpp_prefill_kwargs: dict,
    ):
        """Iter-0 hybrid prefill: compiled torch logits (+ optional torch KV write)."""
        from infinilm.compile.env import prefill_share_weights_enabled

        self._ensure_compiled_prefill_runner()
        runner = self._compiled_prefill_runner
        if runner is None or not self._hybrid_prefill_ready():
            infinicore.sync_device()
            return self(**cpp_prefill_kwargs)

        if input_ids_torch is None:
            input_ids_torch = self._infinicore_input_ids_to_torch(input_ids)
        if prefill_share_weights_enabled():
            kv_layers = self._get_paged_kv_layers()
            last_logits = runner.run_prefill_paged(
                input_ids_torch,
                kv_layers=kv_layers,
                slot_mapping=slot_mapping,
                block_size=paged_block_size,
            )
        else:
            last_logits = runner.run_prefill_last_logits(input_ids_torch)
            infinicore.sync_device()
            self(**cpp_prefill_kwargs)

        infinicore.sync_stream()
        gen_cfg = GenerationConfig(
            max_new_tokens=1,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        return self._sample_token_id_from_logits(last_logits, gen_cfg)

    @property
    def dtype(self):
        torch_dtype = self.hf_config.get("torch_dtype")
        if torch_dtype is None:
            torch_dtype = self.hf_config.get("dtype")
        return parse_dtype(torch_dtype)

    @property
    def model_type(self):
        return self.hf_config["model_type"]

    @property
    def eos_token_id(self):
        # HuggingFace priority: generation_config.json > config.json
        # HuggingFace's documented loading priority for generation parameters
        # (see transformers/generation/utils.py, GenerationMixin.generate docstring):
        #   1) from the `generation_config.json` model file, if it exists
        #   2) from the model configuration (config.json)
        #
        # config.json may contain incomplete or outdated generation parameters
        # because HuggingFace treats config.json as model architecture config
        # and generation_config.json as generation behavior config. For example,
        # InternLM3's config.json has eos_token_id=2, while
        # generation_config.json has eos_token_id=[2, 128131].
        # Following this priority ensures we always get the authoritative value.
        eos_token_id = (
            self.hf_generation_config.get("eos_token_id")
            or self.hf_config.get("eos_token_id")
            or []
        )
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        return eos_token_id

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        input_ids,
        *,
        pixel_values=None,
        position_ids=None,
        past_kv_lengths=None,
        total_kv_lengths=None,
        input_offsets=None,
        cu_seqlens=None,
        block_tables=None,
        slot_mapping=None,
        image_bound=None,
        tgt_sizes=None,
        temperature=None,
        top_k=None,
        top_p=None,
        return_logits=False,
    ):
        try:
            # TODO: Remove `_underlying` and simplify the corresponding code.
            input_ids = input_ids._underlying if input_ids is not None else None
            pixel_values = (
                pixel_values._underlying if pixel_values is not None else None
            )
            position_ids = (
                position_ids._underlying if position_ids is not None else None
            )
            past_kv_lengths = (
                past_kv_lengths._underlying if past_kv_lengths is not None else None
            )
            total_kv_lengths = (
                total_kv_lengths._underlying if total_kv_lengths is not None else None
            )
            input_offsets = (
                input_offsets._underlying if input_offsets is not None else None
            )
            block_tables = (
                block_tables._underlying if block_tables is not None else None
            )
            cu_seqlens = cu_seqlens._underlying if cu_seqlens is not None else None
            slot_mapping = (
                slot_mapping._underlying if slot_mapping is not None else None
            )
            image_bound = image_bound._underlying if image_bound is not None else None
            tgt_sizes = tgt_sizes._underlying if tgt_sizes is not None else None

            input_kwargs = dict(
                input_ids=input_ids,
                position_ids=position_ids,
                past_sequence_lengths=past_kv_lengths,
                total_sequence_lengths=total_kv_lengths,
                input_offsets=input_offsets,
                cu_seqlens=cu_seqlens,
                block_tables=block_tables,
                slot_mapping=slot_mapping,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                return_logits=return_logits,
            )
            if pixel_values is not None:
                input_kwargs["pixel_values"] = pixel_values
            if image_bound is not None:
                input_kwargs["image_bound"] = image_bound
            if tgt_sizes is not None:
                input_kwargs["tgt_sizes"] = tgt_sizes

            output = super().forward(super().Input(**input_kwargs))
            if return_logits:
                if output.logits is None:
                    raise RuntimeError(
                        "InferEngine forward: return_logits=True but C++ returned no logits"
                    )
                return infinicore.Tensor(output.logits)
            return infinicore.Tensor(output.output_ids)
        except BaseException as e:
            handle_oom_and_exit(e)
            raise

    def generate(
        self,
        input_ids,
        generation_config,
        *,
        pixel_values=None,
        image_bound=None,
        tgt_sizes=None,
        _measure_and_log_time=False,
    ):
        eos_token_id = self.eos_token_id

        past_seq_len = 0
        output_ids = []
        initial_batch_size, initial_seqlen = input_ids.shape[:2]
        seq_len = initial_seqlen
        batch_size = initial_batch_size

        if batch_size != 1 and generation_config.max_new_tokens is None:
            raise ValueError(
                "When `batch_size > 1`, `max_new_tokens` must be specified."
            )

        if _measure_and_log_time:
            time_measurements = []

        # Init compiled torch backbone before paged-attn slot/block tensors are
        # allocated (extra GPU allocations during meta-init have caused MACA segfaults).
        if self._compiled_prefill_supported() and self.enable_paged_attn:
            self._ensure_compiled_prefill_runner()

        block_tables = None
        max_blocks_per_batch = 0
        if self.enable_paged_attn:
            paged_block_size = self.get_cache_config().block_size()
            max_blocks_per_batch = (
                initial_seqlen + generation_config.max_new_tokens + paged_block_size - 1
            ) // paged_block_size

            block_tables_list = [
                range(i * max_blocks_per_batch, (i + 1) * max_blocks_per_batch)
                for i in range(batch_size)
            ]
            block_tables = infinicore.from_list(
                block_tables_list,
                dtype=infinicore.int32,
            )

        for iter in range(0, generation_config.max_new_tokens):
            if _measure_and_log_time:
                start_time = time.perf_counter()

            batch_size, seq_len = input_ids.shape[:2]

            if self.enable_paged_attn:
                input_ids = input_ids.view([1, batch_size * seq_len])
                position_ids = infinicore.from_list(
                    list(range(past_seq_len, past_seq_len + seq_len)) * batch_size,
                    dtype=infinicore.int64,
                )

                if iter == 0:
                    slot_mapping_list = []
                    for b in range(batch_size):
                        slot_mapping_list.extend(
                            [
                                b * max_blocks_per_batch * paged_block_size + i
                                for i in range(seq_len)
                            ]
                        )
                else:
                    slot_mapping_list = [
                        i
                        for i in range(
                            past_seq_len,
                            max_blocks_per_batch
                            * paged_block_size
                            * initial_batch_size,
                            max_blocks_per_batch * paged_block_size,
                        )
                    ]

                slot_mapping = infinicore.from_list(
                    slot_mapping_list,
                    dtype=infinicore.int64,
                )
            else:
                position_ids = infinicore.from_list(
                    [
                        list(range(past_seq_len, past_seq_len + seq_len))
                        for _ in range(batch_size)
                    ],
                    dtype=infinicore.int64,
                )

                slot_mapping = None

            past_kv_lengths = infinicore.from_list(
                [past_seq_len] * batch_size, dtype=infinicore.int32
            )
            total_kv_lengths = infinicore.from_list(
                [past_seq_len + seq_len] * batch_size, dtype=infinicore.int32
            )
            cu_seqlens = infinicore.from_list(
                [(past_seq_len + seq_len) * i for i in range(batch_size + 1)],
                dtype=infinicore.int32,
            )
            input_offsets = infinicore.from_list(
                [seq_len * i for i in range(batch_size + 1)], dtype=infinicore.int32
            )

            if (
                iter == 0
                and self._compiled_prefill_supported()
                and self.enable_paged_attn
            ):
                cpp_prefill_kwargs = dict(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    position_ids=position_ids,
                    past_kv_lengths=past_kv_lengths,
                    total_kv_lengths=total_kv_lengths,
                    input_offsets=input_offsets,
                    cu_seqlens=cu_seqlens,
                    block_tables=block_tables,
                    slot_mapping=slot_mapping,
                    image_bound=image_bound,
                    tgt_sizes=tgt_sizes,
                    temperature=generation_config.temperature,
                    top_k=generation_config.top_k,
                    top_p=generation_config.top_p,
                )
                token_id = self._hybrid_compiled_prefill_step(
                    input_ids,
                    temperature=generation_config.temperature,
                    top_k=generation_config.top_k,
                    top_p=generation_config.top_p,
                    block_tables=block_tables,
                    slot_mapping=slot_mapping,
                    paged_block_size=paged_block_size,
                    cpp_prefill_kwargs=cpp_prefill_kwargs,
                )
                output_id = infinicore.from_list([token_id], dtype=infinicore.int64)
            else:
                output_id = self(
                    input_ids=input_ids,
                    pixel_values=pixel_values if iter == 0 else None,
                    position_ids=position_ids,
                    past_kv_lengths=past_kv_lengths,
                    total_kv_lengths=total_kv_lengths,
                    input_offsets=input_offsets,
                    cu_seqlens=cu_seqlens,
                    block_tables=block_tables,
                    slot_mapping=slot_mapping,
                    image_bound=image_bound if iter == 0 else None,
                    tgt_sizes=tgt_sizes if iter == 0 else None,
                    temperature=generation_config.temperature,
                    top_k=generation_config.top_k,
                    top_p=generation_config.top_p,
                )

            output_ids.append(output_id)

            if (
                initial_batch_size == 1
                and generation_config.stop_on_eos
                and generation_config.max_new_tokens is not None
                and int(output_id.to_numpy().reshape(-1)[0]) in eos_token_id
            ):
                break

            # start_prepare_time = time.perf_counter()
            input_ids = output_id.view([batch_size, 1])

            past_seq_len = past_seq_len + seq_len

            if _measure_and_log_time:
                end_time = time.perf_counter()

                time_measurements.append((end_time - start_time))

        if _measure_and_log_time:
            print(
                f"\n\n\n Generation completed in {round(sum(time_measurements) * 1000, 2)} ms"
            )
            print(
                f" Batchsize={initial_batch_size}  Per_Batch_Input_Len={initial_seqlen}  Per_Batch_New_Tokens={len(time_measurements)}\n"
            )
            print(
                f" Prefill TTFT: {round(time_measurements[0] * 1000, 2)} ms  Throughput: {round((initial_batch_size * initial_seqlen) / time_measurements[0], 2)} tok/s\n",
            )
            if len(time_measurements) > 1:
                print(
                    f" Decode  Avg ITL: {round(sum(time_measurements[1:]) * 1000 / (len(time_measurements) - 1), 2)} ms   Throughput: {round((initial_batch_size * (len(time_measurements) - 1)) / sum(time_measurements[1:]), 2)} tok/s\n",
                )

        return output_ids

    def reset_cache(self, cache_config):
        infinicore.sync_device()
        self._paged_kv_layers_cache = None
        self.enable_paged_attn = isinstance(cache_config, PagedKVCacheConfig)
        paged_block_size = (
            cache_config.block_size()
            if isinstance(cache_config, PagedKVCacheConfig)
            else 256
        )
        # Bootstrap compiled backbone before C++ paged KV allocation (MACA segfault workaround).
        if self._compiled_prefill_supported() and self.enable_paged_attn:
            self._ensure_compiled_prefill_runner(block_size=paged_block_size)
        super().reset_cache(cache_config)
        if self._compiled_prefill_runner is not None and self.enable_paged_attn:
            from infinilm.compile.env import (
                prefill_cudagraph_enabled,
                prefill_share_weights_enabled,
            )

            if prefill_share_weights_enabled() and prefill_cudagraph_enabled():
                kv_layers = self._get_paged_kv_layers()
                block_size = self.get_cache_config().block_size()
                self._compiled_prefill_runner.ensure_cudagraph_capture(
                    kv_layers, block_size=block_size
                )
                logger.info(
                    "compiled prefill: CUDAGraph capture complete (ready for hybrid prefill)"
                )

    def state_dict_keyname(self):
        return super().state_dict()[0].keys()

    def load_state_dict(self, state_dict, strict=None):
        for name, param in state_dict.items():
            super().load_param(name, param._underlying)
            
    def process_weights_after_loading(self):
        fn = getattr(super(), "process_weights_after_loading", None)
        if fn is not None:
            fn()
