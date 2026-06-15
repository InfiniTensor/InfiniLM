import logging

from .processor import InfinilmProcessor, register_processor
from transformers import AutoTokenizer
from ..llm.static_scheduler import StaticSchedulerOutput
from ..llm.scheduler import ScheduledRow, SchedulerOutput

logger = logging.getLogger(__name__)


@register_processor("default")
class BasicLLMProcessor(InfinilmProcessor):
    def __init__(self, model_dir_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir_path, trust_remote_code=True
        )
        # Paged KV cache block count; LLMEngine sets this for stable block_tables width.
        self.num_blocks: int | None = None

    @staticmethod
    def _pad_block_table(block_table: list, target_width: int) -> list:
        if target_width <= 0:
            raise ValueError(f"block_tables width must be positive, got {target_width}")
        if len(block_table) > target_width:
            raise ValueError(
                f"block_table length {len(block_table)} exceeds num_blocks {target_width}"
            )
        return block_table + [-1] * (target_width - len(block_table))

    def _block_table_width(self, rows_or_requests) -> int:
        """Pad block_tables to num_blocks so runtime width matches PagedCompiler capture."""
        max_in_batch = max(len(req.block_table) for req in rows_or_requests)
        num_blocks = getattr(self, "num_blocks", None)
        if num_blocks is not None:
            if max_in_batch > num_blocks:
                raise ValueError(
                    f"block_table length {max_in_batch} exceeds num_blocks {num_blocks}"
                )
            return num_blocks
        return max_in_batch

    @staticmethod
    def _slot_mapping_for_hybrid_prefill(slot_mapping: list[int]):
        """GPU slot indices for hybrid CG replay (avoids CPU ``from_list`` on step thread)."""
        import infinicore
        import torch

        if not slot_mapping:
            return infinicore.from_torch(
                torch.empty(0, dtype=torch.int64, device=torch.device("cuda", 0))
            )
        slots = torch.tensor(
            slot_mapping, dtype=torch.int64, device=torch.device("cuda", 0)
        )
        return infinicore.from_torch(slots.contiguous())

    def __call__(self, prompt: str, return_tensors: str = None, **kwargs) -> dict:
        # add_special_tokens=False Prevent duplicate BOS token for Llama-3/3.1 models.
        # The `prompt` string here is already rendered by `apply_chat_template(tokenize=False)`,
        # which explicitly includes the `<|begin_of_text|>` (BOS) token at the start.
        # Since `LlamaTokenizerFast` defaults to `add_bos_token=True`, calling the tokenizer
        # with the default `add_special_tokens=True` would prepend a second BOS token.
        # This shifts the RoPE positional encodings by 1 and causes greedy decoding outputs
        # to diverge significantly from HuggingFace. We must explicitly disable it.
        if return_tensors is None:
            return self.tokenizer(prompt, add_special_tokens=False)
        elif return_tensors == "infini":
            import infinicore

            result = {}
            for key, tensor in self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).items():
                result[key] = tensor.from_torch(tensor)
            return result

        # "pt" or "np" or "tf".
        return self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)

    def apply_chat_template(
        self,
        conversation,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        **kwargs,
    ):
        normalized_conversation = []
        for message in conversation:
            if isinstance(message["content"], list):
                assert len(message["content"]) == 1, "Only one content item supported in list"
                content_item = message["content"][0]
                assert "type" in content_item and "text" in content_item, "Content dict must have 'type' and 'text' keys"
                normalized_conversation.append(
                    {"role": message["role"], "content": content_item["text"]}
                )
            else:
                normalized_conversation.append(message)
        return self.tokenizer.apply_chat_template(
            conversation=normalized_conversation,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
            **kwargs,
        )

    def build_model_inputs(
        self,
        scheduler_output: SchedulerOutput | StaticSchedulerOutput,
        temperature: float = 1.0,
        top_p: float = 0.8,
        top_k: int = 1,
    ) -> dict:
        """Process a batch of data and return a dictionary of model inputs."""
        if isinstance(scheduler_output, StaticSchedulerOutput):
            return self._build_model_input_from_static_scheduler_output(
                scheduler_output, temperature, top_p, top_k
            )
        elif isinstance(scheduler_output, SchedulerOutput):
            return self._build_model_input_from_batch_scheduler_output(
                scheduler_output, temperature, top_p, top_k
            )
        else:
            raise ValueError(
                "scheduler_output must be an instance of SchedulerOutput or StaticSchedulerOutput"
            )

    def _build_model_input_from_static_scheduler_output(
        self, scheduler_output: StaticSchedulerOutput, temperature, top_p, top_k
    ) -> dict:
        """Construct model inputs for prefill or decode phase.

        Static cache model inputs:

        Prefill phase (with prefix cache reuse):
            - input_ids: Tokens after the cached prefix [1, prompt_length - prefix_hit_len]
            - position_ids: [prefix_hit_len, ..., prompt_length-1]
            - past_kv_lengths: [prefix_hit_len]  (reuse cached prefix)
            - total_kv_lengths: [prompt_length]

        Decode phase:
            - input_ids: Only the last generated token [1, 1]
            - position_ids: [current_position] (position in full sequence)
            - past_kv_lengths: [num_cached_tokens]
            - total_kv_lengths: [total_tokens]
        """
        import infinicore

        """Build model input from static scheduler output."""
        req = scheduler_output.scheduled_requests[0]

        if scheduler_output.is_prefill:
            # Prefill: only send tokens not already in cache
            tokens = req.get_input_tokens()
            prefix_hit_len = scheduler_output.prefix_hit_len
            input_tokens = tokens[prefix_hit_len:]
            input_ids = [input_tokens]
            position_ids = [list(range(prefix_hit_len, len(tokens)))]
            past_kv_len = prefix_hit_len
            total_kv_len = len(tokens)
            input_offsets = [0, len(input_tokens)]
        else:
            # Decode: send only the last generated token
            last_token = req.generated_token_ids[-1]
            current_position = req.get_total_length() - 1
            input_ids = [[last_token]]
            position_ids = [[current_position]]
            past_kv_len = current_position
            total_kv_len = req.get_total_length()
            input_offsets = [0, 1]

        return {
            "input_ids": infinicore.from_list(input_ids, dtype=infinicore.int64),
            "position_ids": infinicore.from_list(position_ids, dtype=infinicore.int64),
            "past_kv_lengths": infinicore.from_list(
                [past_kv_len], dtype=infinicore.int32
            ),
            "total_kv_lengths": infinicore.from_list(
                [total_kv_len], dtype=infinicore.int32
            ),
            "input_offsets": infinicore.from_list(
                input_offsets, dtype=infinicore.int32
            ),
            "cu_seqlens": infinicore.from_list(
                [0, total_kv_len], dtype=infinicore.int32
            ),
            "block_tables": None,
            "slot_mapping": None,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        }

    @staticmethod
    def _gpu_infinicore_int32(values: list[int]):
        """Stage small int32 metadata on GPU for hybrid CG replay."""
        import infinicore
        import torch

        if not values:
            return infinicore.from_list(values, dtype=infinicore.int32)
        tensor = torch.tensor(
            values, dtype=torch.int32, device=torch.device("cuda", 0)
        )
        return infinicore.from_torch(tensor.contiguous())

    @staticmethod
    def _gpu_infinicore_int64(values: list[int]):
        """Stage index metadata on GPU for hybrid CG replay (MetaX step thread)."""
        import infinicore
        import torch

        if not values:
            return infinicore.from_list(values, dtype=infinicore.int64)
        tensor = torch.tensor(
            values, dtype=torch.long, device=torch.device("cuda", 0)
        )
        return infinicore.from_torch(tensor.contiguous())

    @staticmethod
    def _append_prefill_row_metadata(
        row: ScheduledRow,
        *,
        tokens: list,
        seq_lens: list,
        seq_offsets: list,
        slot_mapping: list,
        cached_lens: list,
        position_ids: list,
        cu_seqlens: list,
        is_final_prefill_chunk: list,
    ) -> None:
        req = row.request
        num_cached = req.num_cached_tokens
        req_tokens = req.get_input_tokens()
        q = row.num_scheduled_tokens
        start = req.chunk_prefill_offset if req.chunk_prefill_offset > 0 else num_cached
        end = min(start + q, len(req_tokens))
        tokens_to_compute = req_tokens[start:end]
        tokens.extend(tokens_to_compute)
        total_kv_len = end
        seq_lens.append(total_kv_len)
        current_offset = seq_offsets[-1] + q
        seq_offsets.append(current_offset)
        slot_start = start - num_cached
        slot_end = end - num_cached
        slot_mapping.extend(req.slot_mapping[slot_start:slot_end])
        cached_lens.append(start)
        position_ids.extend(range(start, end))
        cu_seqlens.append(cu_seqlens[-1] + q)
        is_final_prefill_chunk.append(row.is_final_prefill_chunk)

    @staticmethod
    def _append_decode_row_metadata(
        req,
        *,
        tokens: list,
        seq_lens: list,
        seq_offsets: list,
        slot_mapping: list,
        cached_lens: list,
        position_ids: list,
        cu_seqlens: list,
        is_final_prefill_chunk: list,
    ) -> None:
        num_cached = req.num_cached_tokens
        seq_len = req.get_total_length()
        last_token = req.generated_token_ids[-1]
        tokens.append(last_token)
        seq_lens.append(seq_len)
        current_offset = seq_offsets[-1] + 1
        seq_offsets.append(current_offset)
        slot_mapping.extend(req.slot_mapping)
        cached_lens.append(num_cached)
        position_ids.append(seq_len - 1)
        cu_seqlens.append(cu_seqlens[-1] + seq_len)
        is_final_prefill_chunk.append(True)

    def _build_model_input_from_batch_scheduler_output(
        self, scheduler_output: SchedulerOutput, temperature, top_p, top_k
    ) -> dict:
        """Construct model inputs from scheduler output (legacy or row-based v1)."""
        import infinicore

        if not scheduler_output.scheduled_requests:
            raise RuntimeError(
                "build_model_inputs called with empty scheduled_requests"
            )

        if scheduler_output.rows:
            return self._build_model_input_from_rows(
                scheduler_output, temperature, top_p, top_k
            )

        return self._build_model_input_legacy_phase(
            scheduler_output, temperature, top_p, top_k
        )

    def _build_model_input_from_rows(
        self, scheduler_output: SchedulerOutput, temperature, top_p, top_k
    ) -> dict:
        """Per-row varlen metadata for v1 (and future mixed) scheduling."""
        import infinicore

        tokens: list = []
        seq_lens: list = []
        seq_offsets: list = [0]
        block_tables: list = []
        slot_mapping: list = []
        cached_lens: list = []
        position_ids: list = []
        cu_seqlens: list = [0]
        is_final_prefill_chunk: list = []

        block_table_width = self._block_table_width(
            [row.request for row in scheduler_output.rows]
        )

        for row in scheduler_output.rows:
            req = row.request
            if row.is_prefill_row:
                self._append_prefill_row_metadata(
                    row,
                    tokens=tokens,
                    seq_lens=seq_lens,
                    seq_offsets=seq_offsets,
                    slot_mapping=slot_mapping,
                    cached_lens=cached_lens,
                    position_ids=position_ids,
                    cu_seqlens=cu_seqlens,
                    is_final_prefill_chunk=is_final_prefill_chunk,
                )
            else:
                self._append_decode_row_metadata(
                    req,
                    tokens=tokens,
                    seq_lens=seq_lens,
                    seq_offsets=seq_offsets,
                    slot_mapping=slot_mapping,
                    cached_lens=cached_lens,
                    position_ids=position_ids,
                    cu_seqlens=cu_seqlens,
                    is_final_prefill_chunk=is_final_prefill_chunk,
                )
            block_tables.append(
                self._pad_block_table(req.block_table, block_table_width)
            )

        return self._finalize_batch_model_input(
            tokens=tokens,
            seq_lens=seq_lens,
            seq_offsets=seq_offsets,
            block_tables=block_tables,
            slot_mapping=slot_mapping,
            cached_lens=cached_lens,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            is_final_prefill_chunk=is_final_prefill_chunk,
            homogeneous_prefill=scheduler_output.is_homogeneous_prefill(),
            scheduling_mode=scheduler_output.scheduling_mode,
            n_req=len(scheduler_output.rows),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

    def _build_model_input_legacy_phase(
        self, scheduler_output: SchedulerOutput, temperature, top_p, top_k
    ) -> dict:
        """Legacy global is_prefill phase batch builder."""
        import infinicore

        tokens: list = []
        seq_lens: list = []
        seq_offsets: list = [0]
        block_tables: list = []
        slot_mapping: list = []
        cached_lens: list = []
        position_ids: list = []
        cu_seqlens: list = [0]
        is_final_prefill_chunk: list = []

        block_table_width = self._block_table_width(
            scheduler_output.scheduled_requests
        )
        current_offset = 0

        for req in scheduler_output.scheduled_requests:
            num_cached = req.num_cached_tokens
            if scheduler_output.is_prefill:
                req_tokens = req.get_input_tokens()

                if req.is_chunking():
                    start = req.chunk_prefill_offset
                    end = min(start + req.chunk_size, len(req_tokens))
                    tokens_to_compute = req_tokens[start:end]
                    compute_len = len(tokens_to_compute)
                    tokens.extend(tokens_to_compute)
                    total_kv_len = start + compute_len
                    seq_lens.append(total_kv_len)
                    current_offset += compute_len
                    seq_offsets.append(current_offset)
                    slot_start = start - num_cached
                    slot_end = end - num_cached
                    slot_mapping.extend(req.slot_mapping[slot_start:slot_end])
                    cached_lens.append(start)
                    position_ids.extend(range(start, end))
                    cu_seqlens.append(cu_seqlens[-1] + len(tokens_to_compute))
                else:
                    tokens_to_compute = req_tokens[num_cached:]
                    tokens.extend(tokens_to_compute)
                    compute_len = len(tokens_to_compute)
                    seq_len = len(req_tokens)
                    seq_lens.append(seq_len)
                    current_offset += compute_len
                    seq_offsets.append(current_offset)
                    slot_mapping.extend(req.slot_mapping)
                    cached_lens.append(num_cached)
                    position_ids.extend(range(num_cached, num_cached + compute_len))
                    cu_seqlens.append(cu_seqlens[-1] + seq_len)
            else:
                seq_len = req.get_total_length()
                last_token = req.generated_token_ids[-1]
                tokens.append(last_token)
                seq_lens.append(seq_len)
                current_offset += 1
                seq_offsets.append(current_offset)
                slot_mapping.extend(req.slot_mapping)
                cached_lens.append(num_cached)
                position_ids.append(seq_len - 1)
                cu_seqlens.append(cu_seqlens[-1] + seq_len)

            block_tables.append(
                self._pad_block_table(req.block_table, block_table_width)
            )

        if scheduler_output.is_prefill:
            for req in scheduler_output.scheduled_requests:
                if req.chunk_size > 0 and req.is_prefill and req.is_chunking():
                    is_final_prefill_chunk.append(req.chunk_is_last())
                else:
                    is_final_prefill_chunk.append(True)

        return self._finalize_batch_model_input(
            tokens=tokens,
            seq_lens=seq_lens,
            seq_offsets=seq_offsets,
            block_tables=block_tables,
            slot_mapping=slot_mapping,
            cached_lens=cached_lens,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            is_final_prefill_chunk=is_final_prefill_chunk,
            homogeneous_prefill=scheduler_output.is_prefill,
            scheduling_mode=(
                "PREFILL" if scheduler_output.is_prefill else "DECODE"
            ),
            n_req=len(scheduler_output.scheduled_requests),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

    def _finalize_batch_model_input(
        self,
        *,
        tokens: list,
        seq_lens: list,
        seq_offsets: list,
        block_tables: list,
        slot_mapping: list,
        cached_lens: list,
        position_ids: list,
        cu_seqlens: list,
        is_final_prefill_chunk: list,
        homogeneous_prefill: bool,
        scheduling_mode: str,
        n_req: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> dict:
        import infinicore

        hybrid_gpu_metadata = False
        total_compute_len = len(tokens)
        if homogeneous_prefill:
            try:
                from infinilm.compile.env import prefill_compile_enabled

                hybrid_gpu_metadata = (
                    prefill_compile_enabled() and total_compute_len > 0
                )
            except ImportError:
                hybrid_gpu_metadata = False

        model_input: dict = {}
        if hybrid_gpu_metadata:
            model_input["position_ids"] = self._gpu_infinicore_int64(position_ids)
            model_input["past_kv_lengths"] = self._gpu_infinicore_int32(cached_lens)
            model_input["total_kv_lengths"] = self._gpu_infinicore_int32(seq_lens)
            model_input["input_offsets"] = self._gpu_infinicore_int32(seq_offsets)
            model_input["cu_seqlens"] = self._gpu_infinicore_int32(cu_seqlens)
            model_input["block_tables"] = self._gpu_infinicore_int32(block_tables)
        else:
            model_input["position_ids"] = infinicore.from_list(
                position_ids, dtype=infinicore.int64
            )
            model_input["past_kv_lengths"] = infinicore.from_list(
                cached_lens, dtype=infinicore.int32
            )
            model_input["total_kv_lengths"] = infinicore.from_list(
                seq_lens, dtype=infinicore.int32
            )
            model_input["input_offsets"] = infinicore.from_list(
                seq_offsets, dtype=infinicore.int32
            )
            model_input["cu_seqlens"] = infinicore.from_list(
                cu_seqlens, dtype=infinicore.int32
            )
            model_input["block_tables"] = infinicore.from_list(
                block_tables, dtype=infinicore.int32
            )
        model_input.update(
            {
                "slot_mapping": self._slot_mapping_for_hybrid_prefill(slot_mapping),
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "is_final_prefill_chunk": is_final_prefill_chunk,
                "scheduling_mode": scheduling_mode,
            }
        )
        if homogeneous_prefill and hybrid_gpu_metadata:
            try:
                import torch

                input_ids_torch = torch.tensor(
                    tokens,
                    dtype=torch.long,
                    device=torch.device("cuda", 0),
                ).view(1, -1)
                model_input["input_ids_torch"] = input_ids_torch
                model_input["input_ids"] = infinicore.from_torch(
                    input_ids_torch.contiguous()
                )
            except ImportError:
                pass
        if "input_ids" not in model_input:
            model_input["input_ids"] = infinicore.from_list(
                [tokens], dtype=infinicore.int64
            )
        if homogeneous_prefill and hybrid_gpu_metadata:
            n_final = sum(is_final_prefill_chunk)
            logger.info(
                "compiled prefill: build_model_inputs prefill "
                "n_req=%s total_compute_len=%s n_final=%s slot_mapping_len=%s mode=%s",
                n_req,
                total_compute_len,
                n_final,
                len(slot_mapping),
                scheduling_mode,
            )
        return model_input

    def get_tokenizer(self):
        return self.tokenizer
