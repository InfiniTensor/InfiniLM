from typing_extensions import override

import torch
from transformers import AutoConfig, AutoProcessor

from .processor import InfinilmProcessor, register_processor
from ..llm.static_scheduler import StaticSchedulerOutput
from ..llm.scheduler import SchedulerOutput


@register_processor("qwen3_vl")
class Qwen3VLProcessor(InfinilmProcessor):
    def __init__(self, model_dir_path: str):
        self.processor = AutoProcessor.from_pretrained(
            model_dir_path, trust_remote_code=True
        )
        self.tokenizer = self.processor.tokenizer
        self.config = AutoConfig.from_pretrained(model_dir_path, trust_remote_code=True)
        self.image_token_id = self.config.image_token_id
        text_config = getattr(self.config, "text_config", None)
        self.pixel_values_dtype = getattr(text_config, "dtype", None) or getattr(
            text_config, "torch_dtype", None
        )
        if self.pixel_values_dtype is None:
            self.pixel_values_dtype = torch.bfloat16

    @override
    def __call__(
        self,
        prompt,
        images=None,
        videos=None,
        audios=None,
        return_tensors: str = None,
        **kwargs,
    ) -> dict:
        if not images and not videos and not audios:
            return self.tokenizer(prompt, return_tensors=return_tensors, **kwargs)

        processor_kwargs = {"text": [prompt], "return_tensors": "pt", **kwargs}
        if images:
            processor_kwargs["images"] = images
        if videos:
            processor_kwargs["videos"] = videos
        return self.processor(**processor_kwargs)

    @override
    def apply_chat_template(
        self,
        conversation,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        **kwargs,
    ):
        normalized = []
        for msg in conversation:
            content = msg["content"]
            if not isinstance(content, list):
                normalized.append(msg)
                continue

            normalized_content = []
            for item in content:
                if item.get("type") == "text":
                    normalized_content.append(item)
                elif item.get("type") == "image_url":
                    normalized_content.append(
                        {"type": "image", "image": item["image_url"]["url"]}
                    )
                else:
                    normalized_content.append(item)
            normalized.append(
                {"role": msg.get("role", "user"), "content": normalized_content}
            )

        return self.processor.apply_chat_template(
            conversation=normalized,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
            **kwargs,
        )

    @override
    def build_model_inputs(
        self,
        scheduler_output: SchedulerOutput | StaticSchedulerOutput,
        temperature: float = 1.0,
        top_p: float = 0.8,
        top_k: int = 1,
        **kwargs,
    ) -> dict:
        if isinstance(scheduler_output, StaticSchedulerOutput):
            return self._build_static(scheduler_output, temperature, top_p, top_k)
        return self._build_paged(scheduler_output, temperature, top_p, top_k)

    def _append_mm_data(self, mm_data: dict, req_id: int, req, num_cached: int):
        if req.processed_inputs is None or "pixel_values" not in req.processed_inputs:
            return
        if num_cached > 0:
            image_token_positions = [
                i
                for i, token in enumerate(req.prompt_token_ids)
                if token == self.image_token_id
            ]
            if image_token_positions and image_token_positions[-1] < num_cached:
                return

        import infinicore

        pixel_values = req.processed_inputs["pixel_values"].to(self.pixel_values_dtype)
        image_grid_thw = req.processed_inputs["image_grid_thw"].to(torch.int64)

        mm_data.setdefault("pixel_values", []).append(
            infinicore.from_torch(pixel_values)
        )
        mm_data.setdefault("tgt_sizes", []).append(
            infinicore.from_torch(image_grid_thw)
        )
        mm_data.setdefault("image_req_ids", []).append(req_id)

    def _build_static(self, scheduler_output, temperature, top_p, top_k):
        import infinicore

        req = scheduler_output.scheduled_requests[0]
        mm_data = {}
        if scheduler_output.is_prefill:
            tokens = req.get_input_tokens()
            prefix_hit_len = scheduler_output.prefix_hit_len
            input_tokens = tokens[prefix_hit_len:]
            position_ids = [list(range(prefix_hit_len, len(tokens)))]
            past_kv_len = prefix_hit_len
            total_kv_len = len(tokens)
            input_offsets = [0, len(input_tokens)]
            self._append_mm_data(mm_data, 0, req, prefix_hit_len)
        else:
            last_token = req.generated_token_ids[-1]
            current_position = req.get_total_length() - 1
            input_tokens = [last_token]
            position_ids = [[current_position]]
            past_kv_len = current_position
            total_kv_len = req.get_total_length()
            input_offsets = [0, 1]

        return {
            "input_ids": infinicore.from_list([input_tokens], dtype=infinicore.int64),
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
            **mm_data,
        }

    def _build_paged(self, scheduler_output, temperature, top_p, top_k):
        import infinicore

        if not scheduler_output.scheduled_requests:
            raise RuntimeError(
                "build_model_inputs called with empty scheduled_requests"
            )

        tokens = []
        seq_lens = []
        seq_offsets = [0]
        block_tables = []
        slot_mapping = []
        cached_lens = []
        position_ids = []
        cu_seqlens = [0]
        mm_data = {}

        max_block_table_len = max(
            len(req.block_table) for req in scheduler_output.scheduled_requests
        )
        current_offset = 0

        for req_id, req in enumerate(scheduler_output.scheduled_requests):
            num_cached = req.num_cached_tokens
            if scheduler_output.is_prefill:
                req_tokens = req.get_input_tokens()
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
                self._append_mm_data(mm_data, req_id, req, num_cached)
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

            block_tables.append(
                req.block_table + [-1] * (max_block_table_len - len(req.block_table))
            )
            cu_seqlens.append(cu_seqlens[-1] + seq_len)

        return {
            "input_ids": infinicore.from_list([tokens], dtype=infinicore.int64),
            "position_ids": infinicore.from_list(position_ids, dtype=infinicore.int64),
            "past_kv_lengths": infinicore.from_list(
                cached_lens, dtype=infinicore.int32
            ),
            "total_kv_lengths": infinicore.from_list(seq_lens, dtype=infinicore.int32),
            "input_offsets": infinicore.from_list(seq_offsets, dtype=infinicore.int32),
            "cu_seqlens": infinicore.from_list(cu_seqlens, dtype=infinicore.int32),
            "block_tables": infinicore.from_list(block_tables, dtype=infinicore.int32),
            "slot_mapping": infinicore.from_list(slot_mapping, dtype=infinicore.int64),
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            **mm_data,
        }

    @override
    def get_tokenizer(self):
        return self.tokenizer

    @override
    def get_mm_token_index_list(
        self, prompt_token_ids, image_ids=None, video_ids=None, **kwargs
    ):
        mappings = []
        idx = 0
        image_idx = 0
        while idx < len(prompt_token_ids):
            if prompt_token_ids[idx] != self.image_token_id:
                idx += 1
                continue
            start = idx
            while (
                idx < len(prompt_token_ids)
                and prompt_token_ids[idx] == self.image_token_id
            ):
                idx += 1
            mappings.append(
                {
                    "start_index": start,
                    "end_index": idx,
                    "identifier": (
                        image_ids[image_idx]
                        if image_ids and image_idx < len(image_ids)
                        else image_idx
                    ),
                }
            )
            image_idx += 1
        return mappings
