from transformers import AutoConfig, AutoProcessor
from typing_extensions import override

from infinilm.multimodal.features import (
    MMFeature,
    MMFeaturePart,
    MMPlaceholderRange,
    iter_mm_parts,
)

from .processor import InfinilmProcessor, register_processor


@register_processor("minicpmv")
class MiniCPMVProcessor(InfinilmProcessor):
    def __init__(self, model_dir_path: str):
        """Initialize the processor with the model directory path."""
        self.processor = AutoProcessor.from_pretrained(
            model_dir_path, trust_remote_code=True
        )
        self.tokenizer = self.processor.tokenizer
        self.config = AutoConfig.from_pretrained(model_dir_path, trust_remote_code=True)
        self.pixel_values_dtype = (
            self.config.dtype
            if hasattr(self.config, "dtype")
            else self.config.torch_dtype
        )

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
        """
        Process the input prompt and media into final inputs.

        {
            'input_ids': TensorShape(1, seq_len),
            'attention_mask': TensorShape(1, seq_len),
            'pixel_values': [[TensorShape(patch_channel, patch_height, patch_width * dim) * n_patches]],
            'image_sizes': [[TensorShape(2,) * n_images]],
            'image_bound': [TensorShape(total_patch, 2)],
            'tgt_sizes': [TensorShape(total_patch, 2)],
        }

        For text-only input, result only contains 'input_ids' and 'attention_mask'.
        """
        if not images and not videos and not audios:
            return self.tokenizer(prompt, return_tensors=return_tensors, **kwargs)

        results = self.processor(
            prompt, images=images, return_tensors="pt", max_slice_nums=9, **kwargs
        )

        return results

    @override
    def apply_chat_template(
        self,
        conversation,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        **kwargs,
    ):
        """Apply chant template given input messages"""
        processed_msg = []
        for msg in conversation:
            content = msg["content"]
            if not isinstance(content, list):
                if isinstance(content, str):
                    processed_msg.append(
                        {"role": msg.get("role", "user"), "content": content}
                    )
                else:
                    raise ValueError("Content must be a list of items or a string")
                continue

            processed_content = []
            for item in content:
                if item.get("type") == "text":
                    processed_content.append(item.get("text", ""))
                elif item.get("type") == "image_url":
                    processed_content.append("(<image>./</image>)")
                else:
                    raise NotImplementedError("Only image input is supported for now")

            processed_msg.append(
                {
                    "role": msg.get("role", "user"),
                    "content": "\n".join(processed_content),
                }
            )

        return self.tokenizer.apply_chat_template(
            conversation=processed_msg,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
            **kwargs,
        )

    @staticmethod
    def _append_mm_data(mm_data: dict, key: str, value):
        mm_data.setdefault(key, []).append(value)

    @staticmethod
    def _flatten_pixel_values(pixel_values):
        flat = []
        if pixel_values is None:
            return flat
        if not isinstance(pixel_values, (list, tuple)):
            return [pixel_values]
        for item in pixel_values:
            if isinstance(item, (list, tuple)):
                flat.extend(item)
            else:
                flat.append(item)
        return flat

    @staticmethod
    def _flatten_tensor_rows(values):
        import torch

        flat = []

        def visit(value):
            if value is None:
                return
            if isinstance(value, torch.Tensor):
                if value.ndim <= 1:
                    flat.append(value)
                else:
                    flat.extend(row for row in value.reshape(-1, value.shape[-1]))
                return
            if isinstance(value, (list, tuple)):
                for item in value:
                    visit(item)
                return
            raise TypeError(f"Unsupported multimodal tensor row type: {type(value)}")

        visit(values)
        return flat

    def _append_request_mm_data(
        self,
        mm_data: dict,
        req,
        req_batch_index: int,
        compute_start: int,
        compute_end: int,
    ) -> None:
        if req.processed_inputs is None or "pixel_values" not in req.processed_inputs:
            return

        import torch
        import infinicore

        flat_pixel_values = self._flatten_pixel_values(
            req.processed_inputs.get("pixel_values")
        )
        flat_tgt_sizes = self._flatten_tensor_rows(
            req.processed_inputs.get("tgt_sizes")
        )

        selected = []
        for feature, part in iter_mm_parts(req.get_mm_features()):
            if feature.modality != "image":
                continue
            copy_start = max(part.token_start, compute_start)
            copy_end = min(part.token_end, compute_end)
            if copy_start >= copy_end:
                continue

            data_index = part.data_index if part.data_index is not None else part.part_index
            if data_index >= len(flat_pixel_values):
                raise IndexError(
                    f"MM part data_index={data_index} exceeds pixel_values "
                    f"count {len(flat_pixel_values)}"
                )
            if data_index >= len(flat_tgt_sizes):
                raise IndexError(
                    f"MM part data_index={data_index} exceeds tgt_sizes "
                    f"count {len(flat_tgt_sizes)}"
                )

            dst_start = copy_start - compute_start
            dst_end = copy_end - compute_start
            src_start = part.embed_start + (copy_start - part.token_start)
            src_end = part.embed_start + (copy_end - part.token_start)
            if (dst_end - dst_start) != (src_end - src_start):
                raise ValueError(
                    "MiniCPM-V multimodal source and destination lengths differ: "
                    f"dst=[{dst_start},{dst_end}), src=[{src_start},{src_end})"
                )

            selected.append((data_index, dst_start, dst_end, src_start, src_end))

        if not selected:
            return

        selected_pixel_values = [
            flat_pixel_values[data_index]
            .flatten(end_dim=1)
            .permute(1, 0)
            .to(self.pixel_values_dtype)
            for data_index, _, _, _, _ in selected
        ]
        pixel_values_tensor = torch.nn.utils.rnn.pad_sequence(
            selected_pixel_values, batch_first=True, padding_value=0.0
        )
        batch_size, seq_len, _ = pixel_values_tensor.shape
        pixel_values_tensor = (
            pixel_values_tensor.permute(0, 2, 1)
            .reshape(batch_size, 3, -1, seq_len)
            .contiguous()
        )

        selected_tgt_sizes = [
            flat_tgt_sizes[data_index].reshape(-1)
            for data_index, _, _, _, _ in selected
        ]
        for tgt_size in selected_tgt_sizes:
            if tgt_size.numel() != 2:
                raise ValueError(
                    "MiniCPM-V tgt_sizes row must contain exactly two values, "
                    f"got {tgt_size.numel()}"
                )
        tgt_sizes_tensor = torch.vstack(selected_tgt_sizes).to(torch.int64)
        image_bound_tensor = torch.tensor(
            [[dst_start, dst_end] for _, dst_start, dst_end, _, _ in selected],
            dtype=torch.int64,
        ).unsqueeze(0)
        image_embed_bound_tensor = torch.tensor(
            [[src_start, src_end] for _, _, _, src_start, src_end in selected],
            dtype=torch.int64,
        ).unsqueeze(0)

        self._append_mm_data(
            mm_data, "pixel_values", infinicore.from_torch(pixel_values_tensor)
        )
        self._append_mm_data(
            mm_data, "tgt_sizes", infinicore.from_torch(tgt_sizes_tensor)
        )
        self._append_mm_data(
            mm_data, "image_bound", infinicore.from_torch(image_bound_tensor)
        )
        self._append_mm_data(
            mm_data,
            "image_embed_bound",
            infinicore.from_torch(image_embed_bound_tensor),
        )
        self._append_mm_data(mm_data, "image_req_ids", req_batch_index)

    @override
    def build_model_inputs(
        self,
        scheduler_output,
        temperature: float = 1.0,
        top_p: float = 0.8,
        top_k: int = 1,
        **kwargs,
    ) -> dict:
        """Build batched infinilm model inputs from the scheduler output."""
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
            if scheduler_output.is_prefill:
                # Prefill phase
                req_tokens = req.get_input_tokens()
                compute_start, compute_end = req.get_scheduled_prefill_window()
                tokens_to_compute = req_tokens[compute_start:compute_end]
                tokens.extend(tokens_to_compute)

                compute_len = len(tokens_to_compute)
                seq_len = len(req_tokens)
                seq_lens.append(seq_len)

                current_offset += compute_len
                seq_offsets.append(current_offset)

                slot_mapping.extend(req.slot_mapping)
                cached_lens.append(compute_start)
                position_ids.extend(range(compute_start, compute_start + compute_len))
                self._append_request_mm_data(
                    mm_data, req, req_id, compute_start, compute_end
                )

            else:
                # Decode phase
                seq_len = req.get_total_length()
                last_token = (
                    req.generated_token_ids[-1]
                    if req.generated_token_ids
                    else req.prompt_token_ids[-1]
                )
                tokens.append(last_token)
                seq_lens.append(seq_len)

                current_offset += 1
                seq_offsets.append(current_offset)

                slot_mapping.extend(req.slot_mapping)
                cached_lens.append(req.num_local_cached_tokens)
                position_ids.append(seq_len - 1)

            # Pad block_table to same length
            padded_block_table = req.block_table + [-1] * (
                max_block_table_len - len(req.block_table)
            )
            block_tables.append(padded_block_table)
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
        """Return the text tokenizer associated with this processor."""
        return self.tokenizer

    @override
    def get_mm_token_index_list(
        self, prompt_token_ids, image_ids=None, video_ids=None, **kwargs
    ):
        image_ids = image_ids or []
        image_idx = -1
        data_index = 0
        patch_start = None
        part_index = 0
        mm_token_index_list = []
        for i, token_id in enumerate(prompt_token_ids):
            if token_id == self.tokenizer.im_id_start_id:
                image_idx += 1
                part_index = 0
            elif token_id in (
                self.tokenizer.im_start_id,
                self.tokenizer.slice_start_id,
            ):
                if image_idx < 0:
                    image_idx = 0
                assert patch_start is None, (
                    "Invalid prompt format: nested MiniCPM-V image placeholder"
                )
                patch_start = i + 1
            elif token_id in (
                self.tokenizer.im_end_id,
                self.tokenizer.slice_end_id,
            ):
                assert patch_start is not None, (
                    "Invalid prompt format: MiniCPM-V image placeholder end found "
                    "before start"
                )
                mm_token_index_list.append(
                    {
                        "start_index": patch_start,
                        "end_index": i - 1,
                        "identifier": image_ids[image_idx],
                        "modality": "image",
                        "item_index": image_idx,
                        "part_index": part_index,
                        "data_index": data_index,
                    }
                )
                data_index += 1
                part_index += 1
                patch_start = None

        assert patch_start is None, "Invalid prompt format: unclosed image placeholder"
        assert image_idx + 1 == len(image_ids), (
            "The number of image tokens does not match the number of images data provided"
        )
        return mm_token_index_list

    @override
    def get_mm_features(
        self,
        prompt_token_ids,
        processed_inputs=None,
        image_ids=None,
        video_ids=None,
        audio_ids=None,
        **kwargs,
    ):
        mappings = self.get_mm_token_index_list(
            prompt_token_ids,
            image_ids=image_ids,
            video_ids=video_ids,
            audio_ids=audio_ids,
            **kwargs,
        )
        image_bound_rows = []
        if processed_inputs is not None and "image_bound" in processed_inputs:
            image_bound_rows = self._flatten_tensor_rows(processed_inputs["image_bound"])
            if len(image_bound_rows) != len(mappings):
                raise ValueError(
                    "MiniCPM-V image_bound rows do not match placeholder ranges: "
                    f"{len(image_bound_rows)} != {len(mappings)}"
                )

        features = []
        for data_index, mapping in enumerate(mappings):
            start = int(mapping["start_index"])
            end = int(mapping["end_index"]) + 1
            if image_bound_rows:
                bound = image_bound_rows[data_index]
                bound_start = int(bound[0].item())
                bound_end = int(bound[1].item())
                if (bound_start, bound_end) != (start, end):
                    raise ValueError(
                        "MiniCPM-V token placeholder range does not match "
                        f"image_bound row {data_index}: token=[{start},{end}), "
                        f"image_bound=[{bound_start},{bound_end})"
                    )

            part = MMFeaturePart(
                token_start=start,
                token_end=end,
                embed_start=0,
                embed_end=end - start,
                item_index=int(mapping["item_index"]),
                part_index=int(mapping["part_index"]),
                data_index=int(mapping.get("data_index", data_index)),
            )
            features.append(
                MMFeature(
                    modality="image",
                    identifier=str(mapping["identifier"]),
                    position=MMPlaceholderRange(offset=start, length=end - start),
                    parts=(part,),
                )
            )
        return features
