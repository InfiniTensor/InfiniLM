from typing_extensions import override

from transformers import AutoConfig, AutoProcessor

from .processor import InfinilmProcessor
from .processor import register_processor


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
            num_cached = req.num_local_cached_tokens
            if scheduler_output.is_prefill:
                # Prefill phase
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

                if (
                    req.processed_inputs is not None
                    and "pixel_values" in req.processed_inputs
                ):
                    import torch

                    num_cached_patch = (
                        (req.processed_inputs["image_bound"][0][:, 1] <= num_cached)
                        .sum()
                        .item()
                    )

                    # if all patches are already cached, skip processing multimodal inputs and return text-only inputs for this request
                    if num_cached_patch < len(req.processed_inputs["pixel_values"]):
                        # 1. pixel_values
                        all_pixel_values = []
                        pixel_values = req.processed_inputs["pixel_values"]
                        tgt_sizes = req.processed_inputs["tgt_sizes"]
                        image_bound = req.processed_inputs["image_bound"]
                        for pv in pixel_values:
                            all_pixel_values.extend(
                                [
                                    t.flatten(end_dim=1)
                                    .permute(1, 0)
                                    .to(self.pixel_values_dtype)
                                    for i, t in enumerate(pv)
                                    if i >= num_cached_patch
                                ]
                            )

                        pixel_values_tensor = torch.nn.utils.rnn.pad_sequence(
                            all_pixel_values, batch_first=True, padding_value=0.0
                        )
                        B, L, _ = pixel_values_tensor.shape
                        pixel_values_tensor = (
                            pixel_values_tensor.permute(0, 2, 1)
                            .reshape(B, 3, -1, L)
                            .contiguous()
                        )
                        pixel_values_infini = infinicore.from_torch(pixel_values_tensor)

                        # 2. tgt_sizes
                        all_tgt_sizes = [
                            tgt_size
                            for i, tgt_size in enumerate(tgt_sizes)
                            if isinstance(tgt_size, torch.Tensor)
                            and i >= num_cached_patch
                        ]

                        tgt_sizes_tensor = torch.vstack(all_tgt_sizes).to(torch.int64)

                        tgt_sizes_infini = infinicore.from_torch(tgt_sizes_tensor)

                        # 3. image_bound
                        batch_size = len(image_bound)
                        max_ranges = max(len(b) for b in image_bound)

                        bound = torch.zeros(
                            (batch_size, max_ranges, 2), dtype=torch.int64
                        )

                        for i, bnd in enumerate(image_bound):
                            bnd = bnd[num_cached_patch:, :]
                            if len(bnd) > 0:
                                bound[i, : len(bnd), :] = bnd

                        image_bound_infini = infinicore.from_torch(bound)

                        def append_mm_data(mm_data__: dict, key__: str, value__):
                            if mm_data__.get(key__) is None:
                                mm_data[key__] = [value__]
                            else:
                                mm_data[key__].append(value__)

                        append_mm_data(mm_data, "pixel_values", pixel_values_infini)
                        append_mm_data(mm_data, "tgt_sizes", tgt_sizes_infini)
                        append_mm_data(mm_data, "image_bound", image_bound_infini)
                        append_mm_data(mm_data, "image_req_ids", req_id)

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
                cached_lens.append(num_cached)
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
        image_idx = -1
        patch_start = []
        patch_end = []
        mm_token_index_list = []
        for i, token_id in enumerate(prompt_token_ids):
            if token_id == self.tokenizer.im_id_start_id:
                assert len(patch_start) == len(patch_end), (
                    "Invalid prompt format: image start token found before previous image end token is closed"
                )
                # deal with previous image patches
                if patch_start:
                    for start, end in zip(patch_start, patch_end):
                        mm_token_index_list.append(
                            {
                                "start_index": start,
                                "end_index": end,
                                "identifier": image_ids[image_idx],
                            }
                        )
                    # reset patch start and end for next image
                    patch_start = []
                    patch_end = []

                # increment image index for next image
                image_idx += 1
                patch_start.append(i + 1)
            elif token_id == self.tokenizer.slice_start_id:
                patch_start.append(i + 1)
            elif (
                token_id == self.tokenizer.im_id_end_id
                or token_id == self.tokenizer.slice_end_id
            ):
                patch_end.append(i - 1)

        if patch_start:
            for start, end in zip(patch_start, patch_end):
                mm_token_index_list.append(
                    {
                        "start_index": start,
                        "end_index": end,
                        "identifier": image_ids[image_idx],
                    }
                )
        assert image_idx + 1 == len(image_ids), (
            "The number of image tokens does not match the number of images data provided"
        )
        return mm_token_index_list
