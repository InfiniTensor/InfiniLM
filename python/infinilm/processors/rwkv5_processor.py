import ast
from pathlib import Path

import infinicore
from typing_extensions import override

from ..llm.scheduler import SchedulerOutput
from ..llm.static_scheduler import StaticSchedulerOutput
from .mamba_processor import MambaProcessor
from .processor import register_processor


class _TrieNode:
    __slots__ = ("children", "token_id")

    def __init__(self):
        self.children = {}
        self.token_id = None


class RWKVWorldTokenizer:
    """Greedy byte-trie tokenizer used by RWKV World checkpoints."""

    def __init__(self, vocab_path: str | Path):
        self.root = _TrieNode()
        self.id_to_bytes = {}
        with Path(vocab_path).open("r", encoding="utf-8") as vocab_file:
            for line in vocab_file:
                first_space = line.index(" ")
                last_space = line.rindex(" ")
                token_id = int(line[:first_space])
                token = ast.literal_eval(line[first_space:last_space])
                token_bytes = token.encode("utf-8") if isinstance(token, str) else token
                expected_length = int(line[last_space:])
                if not isinstance(token_bytes, bytes) or len(token_bytes) != expected_length:
                    raise ValueError(f"Invalid RWKV vocabulary row for token {token_id}")
                self.id_to_bytes[token_id] = token_bytes
                node = self.root
                for byte in token_bytes:
                    node = node.children.setdefault(byte, _TrieNode())
                node.token_id = token_id

        self.eos_token_id = 0
        self.bos_token_id = 0
        self.pad_token_id = 0
        self.vocab_size = len(self.id_to_bytes)

    def encode(self, text: str, add_special_tokens: bool = False, **kwargs):
        del add_special_tokens, kwargs
        source = text.encode("utf-8")
        token_ids = []
        cursor = 0
        while cursor < len(source):
            node = self.root
            scan = cursor
            best_end = None
            best_id = None
            while scan < len(source) and source[scan] in node.children:
                node = node.children[source[scan]]
                scan += 1
                if node.token_id is not None:
                    best_end = scan
                    best_id = node.token_id
            if best_end is None:
                raise ValueError(f"RWKV vocabulary cannot encode byte at offset {cursor}")
            token_ids.append(best_id)
            cursor = best_end
        return token_ids

    def decode(self, token_ids, skip_special_tokens: bool = False, **kwargs):
        del kwargs
        pieces = []
        for token_id in token_ids:
            token_id = int(token_id)
            # RWKV reserves id 0 for EOS; it has no row in the byte vocabulary.
            if token_id == self.eos_token_id:
                continue
            pieces.append(self.id_to_bytes[token_id])
        return b"".join(pieces).decode("utf-8", errors="replace")


@register_processor("rwkv5")
class RWKV5Processor(MambaProcessor):
    def __init__(self, model_dir_path: str):
        vocab_path = Path(model_dir_path) / "rwkv_vocab_v20230424.txt"
        if not vocab_path.exists():
            raise FileNotFoundError(
                f"RWKV tokenizer vocabulary not found: {vocab_path}"
            )
        self.tokenizer = RWKVWorldTokenizer(vocab_path)

    @override
    def __call__(self, prompt: str, return_tensors: str = None, **kwargs) -> dict:
        del kwargs
        token_ids = self.tokenizer.encode(prompt)
        if return_tensors is None:
            return {"input_ids": token_ids}
        if return_tensors == "pt":
            import torch

            return {"input_ids": torch.tensor([token_ids], dtype=torch.long)}
        if return_tensors == "infini":
            return {
                "input_ids": infinicore.from_list(
                    [token_ids], dtype=infinicore.int64
                )
            }
        raise ValueError(f"Unsupported return_tensors value: {return_tensors}")

    @override
    def apply_chat_template(
        self,
        conversation,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        **kwargs,
    ):
        del kwargs
        sections = []
        for message in conversation:
            content = message.get("content", "")
            if isinstance(content, list):
                content = "".join(
                    str(item.get("text", "")) if isinstance(item, dict) else str(item)
                    for item in content
                )
            role = str(message.get("role", "user")).lower()
            if role == "system":
                sections.append(str(content).strip())
            elif role == "assistant":
                sections.append(f"Assistant: {str(content).strip()}")
            else:
                sections.append(f"User: {str(content).strip()}")
        if add_generation_prompt:
            sections.append("Assistant:")
        rendered = "\n\n".join(sections)
        return self.tokenizer.encode(rendered) if tokenize else rendered

    @override
    def build_model_inputs(
        self,
        scheduler_output: SchedulerOutput | StaticSchedulerOutput,
        temperature: float = 1.0,
        top_p: float = 0.8,
        top_k: int = 1,
        **kwargs,
    ) -> dict:
        model_inputs = super().build_model_inputs(
            scheduler_output,
            temperature,
            top_p,
            top_k,
            **kwargs,
        )

        init_indices = []
        final_indices = []
        for request in scheduler_output.scheduled_requests:
            state_index = request.mamba_cache_index
            if isinstance(scheduler_output, StaticSchedulerOutput):
                state_index = 1
            if state_index is None:
                raise RuntimeError(
                    f"Request {request.request_id} has no assigned RWKV state row"
                )
            init_indices.append(
                0 if scheduler_output.is_prefill else state_index
            )
            final_indices.append(state_index)

        model_inputs["mamba_init_state_indices"] = infinicore.from_list(
            init_indices, dtype=infinicore.int32
        )
        model_inputs["mamba_final_state_indices"] = infinicore.from_list(
            final_indices, dtype=infinicore.int32
        )
        # RWKV5 has no attention KV cache; only its recurrent state is needed.
        model_inputs["block_tables"] = None
        model_inputs["slot_mapping"] = None
        return model_inputs
