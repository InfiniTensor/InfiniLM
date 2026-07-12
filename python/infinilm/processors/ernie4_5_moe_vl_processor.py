from transformers import AutoProcessor
from typing_extensions import override

from .processor import InfinilmProcessor, register_processor


@register_processor("ernie4_5_moe_vl")
class Ernie4_5MoeVLProcessor(InfinilmProcessor):
    def __init__(self, model_dir_path: str):
        self.processor = AutoProcessor.from_pretrained(
            model_dir_path, trust_remote_code=True
        )
        self.tokenizer = self.processor.tokenizer

    @override
    def __call__(
        self,
        prompt=None,
        images=None,
        videos=None,
        audios=None,
        return_tensors: str = None,
        **kwargs,
    ) -> dict:
        if prompt is None:
            prompt = kwargs.pop("text", None)
        if prompt is None:
            raise ValueError("prompt or text must be provided")
        if images is None and videos is None and audios is None:
            return self.tokenizer(prompt, return_tensors=return_tensors, **kwargs)
        return self.processor(
            prompt,
            images=images,
            videos=videos,
            return_tensors=return_tensors or "pt",
            **kwargs,
        )

    @override
    def apply_chat_template(
        self,
        conversation,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        **kwargs,
    ):
        normalized = []
        for message in conversation:
            role = message.get("role", "user")
            content = message.get("content", "")
            if not isinstance(content, list):
                normalized.append({"role": role, "content": content})
                continue

            parts = []
            for item in content:
                item_type = item.get("type")
                if item_type == "text":
                    parts.append(item.get("text", ""))
                elif item_type == "image_url":
                    parts.append(
                        self.processor.IMG_START
                        + "<|image@placeholder|>"
                        + self.processor.IMG_END
                    )
                elif item_type == "video_url":
                    parts.append(
                        self.processor.VID_START
                        + "<|video@placeholder|>"
                        + self.processor.VID_END
                    )
                else:
                    raise NotImplementedError(
                        "Only text, image_url and video_url inputs are supported"
                    )
            normalized.append({"role": role, "content": "".join(parts)})

        return self.tokenizer.apply_chat_template(
            conversation=normalized,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
            **kwargs,
        )

    @override
    def build_model_inputs(
        self,
        scheduler_output,
        temperature: float = 1.0,
        top_p: float = 0.8,
        top_k: int = 1,
        **kwargs,
    ) -> dict:
        import infinicore
        import torch

        if not scheduler_output.scheduled_requests:
            raise RuntimeError(
                "build_model_inputs called with empty scheduled_requests"
            )

        static_prefix_hit = getattr(scheduler_output, "prefix_hit_len", None)
        if static_prefix_hit is not None:
            req = scheduler_output.scheduled_requests[0]
            processed = req.processed_inputs
            has_vision_inputs = (
                processed is not None and processed.get("images") is not None
            )

            if scheduler_output.is_prefill:
                req_tokens = req.get_input_tokens()
                # Static cache does not track multimodal token bounds. Recompute
                # vision prompts from the start so image feature replacement stays
                # aligned with the patch tokens present in input_ids.
                num_cached = 0 if has_vision_inputs else static_prefix_hit
                input_tokens = req_tokens[num_cached:]
                seq_len = len(req_tokens)
                input_offsets = [0, len(input_tokens)]

                if processed is not None and "position_ids" in processed:
                    position_ids = (
                        processed["position_ids"][:, num_cached:seq_len]
                        .to(torch.int64)
                        .contiguous()
                    )
                else:
                    position_ids = [
                        list(range(num_cached, num_cached + len(input_tokens)))
                    ]

                token_type_ids = None
                if processed is not None and "token_type_ids" in processed:
                    token_type_ids = (
                        processed["token_type_ids"][0, num_cached:seq_len]
                        .to(torch.int64)
                        .tolist()
                    )
            else:
                seq_len = req.get_total_length()
                input_tokens = [
                    req.generated_token_ids[-1]
                    if req.generated_token_ids
                    else req.prompt_token_ids[-1]
                ]
                num_cached = seq_len - 1
                input_offsets = [0, 1]

                if processed is not None and "position_ids" in processed:
                    max_position = processed["position_ids"][0].max(dim=0)[0]
                    generated = max_position + (seq_len - len(req.get_input_tokens()))
                    position_ids = generated.to(torch.int64).view(1, 1, 3).contiguous()
                else:
                    position_ids = [[seq_len - 1]]

                token_type_ids = None
                if processed is not None and "token_type_ids" in processed:
                    token_type_ids = [0]

            result = {
                "input_ids": infinicore.from_list(
                    [input_tokens], dtype=infinicore.int64
                ),
                "position_ids": (
                    infinicore.from_torch(position_ids)
                    if torch.is_tensor(position_ids)
                    else infinicore.from_list(position_ids, dtype=infinicore.int64)
                ),
                "past_kv_lengths": infinicore.from_list(
                    [num_cached], dtype=infinicore.int32
                ),
                "total_kv_lengths": infinicore.from_list(
                    [seq_len], dtype=infinicore.int32
                ),
                "input_offsets": infinicore.from_list(
                    input_offsets, dtype=infinicore.int32
                ),
                "cu_seqlens": infinicore.from_list(
                    [0, seq_len], dtype=infinicore.int32
                ),
                "block_tables": None,
                "slot_mapping": None,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
            }
            if token_type_ids is not None:
                result["token_type_ids"] = infinicore.from_list(
                    [token_type_ids], dtype=infinicore.int64
                )
            if scheduler_output.is_prefill and has_vision_inputs:
                result["images"] = infinicore.from_torch(
                    processed["images"].contiguous()
                )
                result["grid_thw"] = infinicore.from_torch(
                    processed["grid_thw"].to(torch.int64).contiguous()
                )
                result["image_type_ids"] = infinicore.from_torch(
                    processed["image_type_ids"].to(torch.int64).contiguous()
                )
                result["image_req_ids"] = [0]
            return result

        tokens = []
        seq_lens = []
        seq_offsets = [0]
        block_tables = []
        slot_mapping = []
        cached_lens = []
        position_ids = []
        token_type_ids = []
        cu_seqlens = [0]
        mm_data = {}

        max_block_table_len = max(
            len(req.block_table) for req in scheduler_output.scheduled_requests
        )
        current_offset = 0

        def append_mm_data(key, value):
            if mm_data.get(key) is None:
                mm_data[key] = [value]
            else:
                mm_data[key].append(value)

        for req_id, req in enumerate(scheduler_output.scheduled_requests):
            processed = req.processed_inputs
            has_vision_inputs = (
                processed is not None and processed.get("images") is not None
            )

            if scheduler_output.is_prefill:
                num_cached = req.num_local_cached_tokens

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

                if processed is not None and "position_ids" in processed:
                    position_ids.extend(
                        processed["position_ids"][0, num_cached:seq_len].tolist()
                    )
                else:
                    position_ids.extend(range(num_cached, num_cached + compute_len))

                if processed is not None and "token_type_ids" in processed:
                    token_type_ids.extend(
                        processed["token_type_ids"][0, num_cached:seq_len]
                        .to(torch.int64)
                        .tolist()
                    )

                if has_vision_inputs and (
                    num_cached == 0
                    or req_tokens[num_cached:].count(self.processor.image_patch_id) > 0
                ):
                    append_mm_data(
                        "images",
                        infinicore.from_torch(processed["images"].contiguous()),
                    )
                    append_mm_data(
                        "grid_thw",
                        infinicore.from_torch(
                            processed["grid_thw"].to(torch.int64).contiguous()
                        ),
                    )
                    append_mm_data(
                        "image_type_ids",
                        infinicore.from_torch(
                            processed["image_type_ids"].to(torch.int64).contiguous()
                        ),
                    )
                    append_mm_data("image_req_ids", req_id)
            else:
                num_cached = req.num_local_cached_tokens
                seq_len = req.get_total_length()
                last_token = req.generated_token_ids[-1]
                tokens.append(last_token)
                seq_lens.append(seq_len)

                current_offset += 1
                seq_offsets.append(current_offset)

                slot_mapping.extend(req.slot_mapping)
                cached_lens.append(num_cached)

                if processed is not None and "position_ids" in processed:
                    max_position = processed["position_ids"][0].max(dim=0)[0]
                    generated = max_position + (seq_len - len(req.get_input_tokens()))
                    position_ids.append(generated.to(torch.int64).tolist())
                else:
                    position_ids.append(seq_len - 1)

                if processed is not None and "token_type_ids" in processed:
                    token_type_ids.append(0)

            padded_block_table = req.block_table + [-1] * (
                max_block_table_len - len(req.block_table)
            )
            block_tables.append(padded_block_table)
            cu_seqlens.append(cu_seqlens[-1] + seq_len)

        position_ids_payload = (
            [position_ids]
            if position_ids and isinstance(position_ids[0], list)
            else position_ids
        )
        position_ids_value = (
            infinicore.from_torch(
                torch.tensor(position_ids_payload, dtype=torch.int64).contiguous()
            )
            if (
                position_ids_payload
                and isinstance(position_ids_payload[0], list)
                and position_ids_payload[0]
                and isinstance(position_ids_payload[0][0], list)
            )
            else infinicore.from_list(position_ids_payload, dtype=infinicore.int64)
        )
        result = {
            "input_ids": infinicore.from_list([tokens], dtype=infinicore.int64),
            "position_ids": position_ids_value,
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
        if token_type_ids:
            result["token_type_ids"] = infinicore.from_list(
                [token_type_ids], dtype=infinicore.int64
            )
        return result

    @override
    def get_tokenizer(self):
        return self.tokenizer

    @override
    def get_mm_token_index_list(
        self, prompt_token_ids, image_ids=None, video_ids=None, audio_ids=None, **kwargs
    ):
        mappings = []
        vision_ids = (image_ids or []) + (video_ids or [])
        image_index = 0
        patch_id = self.processor.image_patch_id
        idx = 0
        prompt_len = len(prompt_token_ids)
        while idx < len(prompt_token_ids):
            if prompt_token_ids[idx] != patch_id:
                idx += 1
                continue
            start = idx
            while idx < len(prompt_token_ids) and prompt_token_ids[idx] == patch_id:
                idx += 1
            identifier = (
                vision_ids[image_index]
                if image_index < len(vision_ids)
                else f"vision_{image_index}"
            )
            mappings.append(
                {
                    "start_index": start,
                    "end_index": prompt_len,
                    "identifier": identifier,
                }
            )
            image_index += 1
        return mappings
