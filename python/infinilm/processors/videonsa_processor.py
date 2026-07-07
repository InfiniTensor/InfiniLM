import os

import torch
from transformers import AutoConfig, AutoProcessor
from typing_extensions import override

from infinilm.multimodal.features import (
    MMFeature,
    MMFeaturePart,
    MMPlaceholderRange,
    iter_mm_parts,
)

from .processor import InfinilmProcessor, register_processor


def decode_video_frames(video_path, num_frames=None):
    if not video_path:
        return None
    try:
        from decord import VideoReader, cpu
        from PIL import Image

        reader = VideoReader(video_path, ctx=cpu(0))
        total = len(reader)
        if total == 0:
            return video_path
        num_frames = max(1, min(num_frames or total, total))
        if num_frames == 1:
            indices = [0]
        else:
            indices = [
                round(i * (total - 1) / (num_frames - 1)) for i in range(num_frames)
            ]
        batch = reader.get_batch(indices).asnumpy()
        return [Image.fromarray(frame) for frame in batch]
    except Exception:
        pass

    try:
        import cv2
        from PIL import Image
    except Exception:
        return video_path

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return video_path
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            return video_path
        num_frames = max(1, min(num_frames or total, total))
        if num_frames == 1:
            indices = [0]
        else:
            indices = [
                round(i * (total - 1) / (num_frames - 1)) for i in range(num_frames)
            ]
        frames = []
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                continue
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        return frames or video_path
    finally:
        cap.release()


@register_processor("videonsa")
class VideoNSAProcessor(InfinilmProcessor):
    def __init__(self, model_dir_path: str):
        self.config = AutoConfig.from_pretrained(model_dir_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(
            model_dir_path, trust_remote_code=True
        )
        self.tokenizer = self.processor.tokenizer
        self.image_token_id = getattr(self.processor, "image_token_id", None)
        if self.image_token_id is None:
            self.image_token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.video_token_id = getattr(self.processor, "video_token_id", None)
        if self.video_token_id is None:
            self.video_token_id = self.tokenizer.convert_tokens_to_ids("<|video_pad|>")
        self.pixel_values_dtype = torch.bfloat16
        self._configure_media_processors_from_env()

    def _configure_media_processors_from_env(self):
        image_processor = getattr(self.processor, "image_processor", None)
        image_max_pixels = os.getenv("INFINILM_VIDEONSA_IMAGE_MAX_PIXELS")
        image_min_pixels = os.getenv("INFINILM_VIDEONSA_IMAGE_MIN_PIXELS")
        if image_processor is not None and image_max_pixels:
            value = int(image_max_pixels)
            image_processor.max_pixels = value
            image_processor.size = dict(getattr(image_processor, "size", {}) or {})
            image_processor.size["longest_edge"] = value
        if image_processor is not None and image_min_pixels:
            value = int(image_min_pixels)
            image_processor.min_pixels = value
            image_processor.size = dict(getattr(image_processor, "size", {}) or {})
            image_processor.size["shortest_edge"] = value

        video_processor = getattr(self.processor, "video_processor", None)
        if video_processor is None:
            return
        num_frames = os.getenv("INFINILM_VIDEONSA_VIDEO_NUM_FRAMES")
        max_pixels = os.getenv("INFINILM_VIDEONSA_VIDEO_MAX_PIXELS")
        min_pixels = os.getenv("INFINILM_VIDEONSA_VIDEO_MIN_PIXELS")
        if num_frames:
            video_processor.num_frames = int(num_frames)
            video_processor.do_sample_frames = True
        if max_pixels:
            value = int(max_pixels)
            video_processor.max_pixels = value
            video_processor.size = dict(getattr(video_processor, "size", {}) or {})
            video_processor.size["longest_edge"] = value
        if min_pixels:
            value = int(min_pixels)
            video_processor.min_pixels = value
            video_processor.size = dict(getattr(video_processor, "size", {}) or {})
            video_processor.size["shortest_edge"] = value

    def _normalize_messages(self, conversation):
        normalized = []
        for msg in conversation:
            content = msg.get("content", [])
            if isinstance(content, str):
                normalized.append({"role": msg.get("role", "user"), "content": content})
                continue
            items = []
            for item in content:
                typ = item.get("type")
                if typ == "text":
                    items.append({"type": "text", "text": item.get("text", "")})
                elif typ == "image_url":
                    items.append({"type": "image", "image": item["image_url"]["url"]})
                elif typ == "video_url":
                    items.append({"type": "video", "video": item["video_url"]["url"]})
                else:
                    raise NotImplementedError(
                        f"Unsupported VideoNSA content type: {typ}"
                    )
            normalized.append({"role": msg.get("role", "user"), "content": items})
        return normalized

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
        results = self.processor(
            text=prompt,
            images=images or None,
            videos=videos or None,
            return_tensors=return_tensors,
            return_mm_token_type_ids=False,
            **kwargs,
        )
        input_ids = results["input_ids"]
        bounds = []
        for row in input_ids:
            row_bounds = []
            start = None
            current = None
            for idx, token_id in enumerate(row.tolist()):
                if token_id in (self.image_token_id, self.video_token_id):
                    if start is None:
                        start = idx
                        current = token_id
                    elif current != token_id:
                        row_bounds.append([start, idx])
                        start = idx
                        current = token_id
                elif start is not None:
                    row_bounds.append([start, idx])
                    start = None
                    current = None
            if start is not None:
                row_bounds.append([start, len(row)])
            bounds.append(row_bounds)

        if bounds and any(bounds):
            max_bounds = max(len(b) for b in bounds)
            image_bound = torch.zeros((len(bounds), max_bounds, 2), dtype=torch.long)
            for i, row_bounds in enumerate(bounds):
                if row_bounds:
                    image_bound[i, : len(row_bounds), :] = torch.tensor(
                        row_bounds, dtype=torch.long
                    )
            results["image_bound"] = image_bound
        return results

    def _valid_media_bounds(self, processed_inputs):
        if processed_inputs is None or "image_bound" not in processed_inputs:
            return []
        bounds = processed_inputs["image_bound"]
        if isinstance(bounds, torch.Tensor):
            bounds = bounds[0].tolist()
        else:
            bounds = bounds[0]
        return [
            (int(start), int(end)) for start, end in bounds if int(end) > int(start)
        ]

    def _grid_list(self, processed_inputs, key):
        value = None if processed_inputs is None else processed_inputs.get(key)
        if value is None:
            return []
        if isinstance(value, torch.Tensor):
            return [tuple(int(x) for x in row.tolist()) for row in value]
        return [tuple(int(x) for x in row) for row in value]

    def _video_second_per_grid(self, processed_inputs, video_idx):
        value = (
            None
            if processed_inputs is None
            else processed_inputs.get("second_per_grid_ts")
        )
        if value is None:
            return 1.0
        if isinstance(value, torch.Tensor):
            value = value.flatten().tolist()
        if isinstance(value, (list, tuple)) and video_idx < len(value):
            return float(value[video_idx])
        try:
            return float(value)
        except Exception:
            return 1.0

    @staticmethod
    def _append_mm_data(mm_data: dict, key: str, value):
        mm_data.setdefault(key, []).append(value)

    def _media_payloads(self, processed_inputs):
        payloads = {"image": [], "video": []}
        if processed_inputs is None:
            return payloads

        pixel_values = processed_inputs.get("pixel_values")
        image_grids = self._grid_list(processed_inputs, "image_grid_thw")
        if pixel_values is not None:
            offset = 0
            for grid in image_grids:
                grid_tensor = torch.as_tensor(grid, dtype=torch.int64)
                patch_count = int(grid_tensor.prod().item())
                payloads["image"].append(
                    (pixel_values[offset : offset + patch_count], grid_tensor)
                )
                offset += patch_count

        pixel_values_videos = processed_inputs.get("pixel_values_videos")
        video_grids = self._grid_list(processed_inputs, "video_grid_thw")
        if pixel_values_videos is not None:
            offset = 0
            for grid in video_grids:
                grid_tensor = torch.as_tensor(grid, dtype=torch.int64)
                patch_count = int(grid_tensor.prod().item())
                payloads["video"].append(
                    (pixel_values_videos[offset : offset + patch_count], grid_tensor)
                )
                offset += patch_count

        return payloads

    def _append_request_mm_data(
        self,
        mm_data: dict,
        req,
        req_batch_index: int,
        compute_start: int,
        compute_end: int,
        packed_request_start: int,
    ) -> None:
        if req.processed_inputs is None or "image_bound" not in req.processed_inputs:
            return

        import infinicore

        payloads = self._media_payloads(req.processed_inputs)
        selected = []

        for feature, part in iter_mm_parts(req.get_mm_features()):
            if feature.modality not in payloads:
                continue

            copy_start = max(part.token_start, compute_start)
            copy_end = min(part.token_end, compute_end)
            if copy_start >= copy_end:
                continue

            item_index = part.item_index
            if item_index >= len(payloads[feature.modality]):
                raise IndexError(
                    f"VideoNSA {feature.modality} item_index={item_index} exceeds "
                    f"processed input count {len(payloads[feature.modality])}"
                )

            dst_start = copy_start - compute_start
            dst_end = copy_end - compute_start
            src_start = part.embed_start + (copy_start - part.token_start)
            src_end = part.embed_start + (copy_end - part.token_start)
            selected.append(
                (
                    feature.modality,
                    payloads[feature.modality][item_index],
                    dst_start,
                    dst_end,
                    src_start,
                    src_end,
                )
            )

        if not selected:
            return

        for modality, payload, dst_start, dst_end, src_start, src_end in selected:
            pixel_tensor = payload[0].to(self.pixel_values_dtype).contiguous()
            grid_tensor = payload[1].reshape(1, 3).to(torch.int64)
            image_bound_tensor = torch.tensor(
                [[[dst_start, dst_end]]],
                dtype=torch.int64,
            )
            image_embed_bound_tensor = torch.tensor(
                [[[src_start, src_end]]],
                dtype=torch.int64,
            )

            self._append_mm_data(
                mm_data, "pixel_values", infinicore.from_torch(pixel_tensor)
            )
            self._append_mm_data(
                mm_data, "tgt_sizes", infinicore.from_torch(grid_tensor)
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

            mm_data.setdefault("visual_token_ranges", []).extend(
                [
                    packed_request_start + int(dst_start),
                    packed_request_start + int(dst_end),
                ]
            )

    def _append_text_positions(self, axes, length, start_pos):
        if length <= 0:
            return start_pos
        vals = list(range(start_pos, start_pos + length))
        axes[0].extend(vals)
        axes[1].extend(vals)
        axes[2].extend(vals)
        return start_pos + length

    def _append_visual_positions(
        self, axes, grid, span_len, start_pos, second_per_grid=1.0
    ):
        spatial_merge_size = int(
            getattr(
                getattr(self.config, "vision_config", None), "spatial_merge_size", 2
            )
        )
        tokens_per_second = float(
            getattr(
                getattr(self.config, "vision_config", None), "tokens_per_second", 1.0
            )
        )
        grid_t, grid_h, grid_w = (int(grid[0]), int(grid[1]), int(grid[2]))
        llm_grid_t = max(1, grid_t)
        llm_grid_h = max(1, grid_h // spatial_merge_size)
        llm_grid_w = max(1, grid_w // spatial_merge_size)
        t_step = max(1, int(round(tokens_per_second * second_per_grid)))

        t_ids, h_ids, w_ids = [], [], []
        for t in range(llm_grid_t):
            for h in range(llm_grid_h):
                for w in range(llm_grid_w):
                    t_ids.append(t * t_step + start_pos)
                    h_ids.append(h + start_pos)
                    w_ids.append(w + start_pos)

        if not t_ids:
            t_ids = h_ids = w_ids = [start_pos]
        if len(t_ids) < span_len:
            repeat = span_len - len(t_ids)
            t_ids.extend([t_ids[-1]] * repeat)
            h_ids.extend([h_ids[-1]] * repeat)
            w_ids.extend([w_ids[-1]] * repeat)
        elif len(t_ids) > span_len:
            t_ids = t_ids[:span_len]
            h_ids = h_ids[:span_len]
            w_ids = w_ids[:span_len]

        axes[0].extend(t_ids)
        axes[1].extend(h_ids)
        axes[2].extend(w_ids)
        return max(max(t_ids), max(h_ids), max(w_ids)) + 1

    def _prompt_mrope_positions(self, token_ids, processed_inputs):
        axes = [[], [], []]
        bounds = self._valid_media_bounds(processed_inputs)
        image_grids = self._grid_list(processed_inputs, "image_grid_thw")
        video_grids = self._grid_list(processed_inputs, "video_grid_thw")
        image_idx = 0
        video_idx = 0
        cursor = 0
        pos_base = 0

        for start, end in bounds:
            if start > len(token_ids):
                break
            end = min(end, len(token_ids))
            pos_base = self._append_text_positions(axes, start - cursor, pos_base)
            token_id = (
                token_ids[start] if start < len(token_ids) else self.image_token_id
            )
            span_len = max(0, end - start)
            if token_id == self.video_token_id and video_idx < len(video_grids):
                pos_base = self._append_visual_positions(
                    axes,
                    video_grids[video_idx],
                    span_len,
                    pos_base,
                    self._video_second_per_grid(processed_inputs, video_idx),
                )
                video_idx += 1
            elif image_idx < len(image_grids):
                pos_base = self._append_visual_positions(
                    axes, image_grids[image_idx], span_len, pos_base
                )
                image_idx += 1
            else:
                pos_base = self._append_text_positions(axes, span_len, pos_base)
            cursor = end

        self._append_text_positions(axes, len(token_ids) - cursor, pos_base)
        return axes

    def _mrope_delta(self, token_ids, processed_inputs):
        positions = self._prompt_mrope_positions(token_ids, processed_inputs)
        if not positions[0]:
            return 0
        return max(max(axis) for axis in positions) + 1 - len(token_ids)

    @override
    def apply_chat_template(
        self,
        conversation,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        **kwargs,
    ):
        return self.processor.apply_chat_template(
            self._normalize_messages(conversation),
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
        position_axes = [[], [], []]
        cu_seqlens = [0]
        mm_data = {}
        max_block_table_len = max(
            1, *(len(req.block_table) for req in scheduler_output.scheduled_requests)
        )
        current_offset = 0

        for req_id, req in enumerate(scheduler_output.scheduled_requests):
            num_cached = req.num_local_cached_tokens
            if scheduler_output.is_prefill:
                req_tokens = req.get_input_tokens()
                compute_start, compute_end = req.get_scheduled_prefill_window()
                tokens_to_compute = req_tokens[compute_start:compute_end]
                tokens.extend(tokens_to_compute)
                compute_len = len(tokens_to_compute)
                seq_len = len(req_tokens)
                seq_lens.append(seq_len)
                packed_request_start = current_offset
                current_offset += compute_len
                seq_offsets.append(current_offset)
                cached_lens.append(compute_start)
                prompt_positions = self._prompt_mrope_positions(
                    req_tokens, req.processed_inputs
                )
                for axis in range(3):
                    position_axes[axis].extend(
                        prompt_positions[axis][compute_start:compute_end]
                    )

                self._append_request_mm_data(
                    mm_data,
                    req,
                    req_id,
                    compute_start,
                    compute_end,
                    packed_request_start,
                )
            else:
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
                cached_lens.append(num_cached)
                mrope_delta = self._mrope_delta(
                    req.prompt_token_ids, req.processed_inputs
                )
                decode_pos = seq_len - 1 + mrope_delta
                for axis in range(3):
                    position_axes[axis].append(decode_pos)

            if req.slot_mapping:
                padded_slot_mapping = req.slot_mapping
            else:
                padded_slot_mapping = (
                    list(range(cu_seqlens[-1], cu_seqlens[-1] + len(tokens_to_compute)))
                    if scheduler_output.is_prefill
                    else [current_offset - 1]
                )
            slot_mapping.extend(padded_slot_mapping)

            padded_block_table = req.block_table + [-1] * (
                max_block_table_len - len(req.block_table)
            )
            block_tables.append(padded_block_table)
            cu_seqlens.append(cu_seqlens[-1] + seq_lens[-1])

        return {
            "input_ids": infinicore.from_list([tokens], dtype=infinicore.int64),
            "position_ids": infinicore.from_list(position_axes, dtype=infinicore.int64),
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
        image_ids = list(image_ids or [])
        video_ids = list(video_ids or [])
        out = []
        image_idx = 0
        video_idx = 0
        data_index = 0
        start = None
        current = None
        current_modality = None
        current_item_index = None

        def append_current(end):
            nonlocal data_index
            if start is None:
                return
            if current_modality == "image":
                identifier = image_ids[current_item_index]
            else:
                identifier = video_ids[current_item_index]
            out.append(
                {
                    "start_index": start,
                    "end_index": end,
                    "end": end,
                    "identifier": identifier,
                    "modality": current_modality,
                    "item_index": current_item_index,
                    "part_index": 0,
                    "data_index": data_index,
                }
            )
            data_index += 1

        for i, token_id in enumerate(prompt_token_ids):
            if token_id in (self.image_token_id, self.video_token_id):
                modality = "image" if token_id == self.image_token_id else "video"
                if start is None:
                    start = i
                    current = token_id
                    current_modality = modality
                    if modality == "image":
                        if image_idx >= len(image_ids):
                            raise ValueError(
                                "The number of image tokens exceeds image data provided"
                            )
                        current_item_index = image_idx
                        image_idx += 1
                    else:
                        if video_idx >= len(video_ids):
                            raise ValueError(
                                "The number of video tokens exceeds video data provided"
                            )
                        current_item_index = video_idx
                        video_idx += 1
                elif current != token_id:
                    append_current(i)
                    start = i
                    current = token_id
                    current_modality = modality
                    if modality == "image":
                        if image_idx >= len(image_ids):
                            raise ValueError(
                                "The number of image tokens exceeds image data provided"
                            )
                        current_item_index = image_idx
                        image_idx += 1
                    else:
                        if video_idx >= len(video_ids):
                            raise ValueError(
                                "The number of video tokens exceeds video data provided"
                            )
                        current_item_index = video_idx
                        video_idx += 1
            elif start is not None:
                append_current(i)
                start = None
                current = None
                current_modality = None
                current_item_index = None
        if start is not None:
            append_current(len(prompt_token_ids))
        if image_idx != len(image_ids):
            raise ValueError(
                "The number of image tokens does not match the number of images data provided"
            )
        if video_idx != len(video_ids):
            raise ValueError(
                "The number of video tokens does not match the number of videos data provided"
            )
        return out

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
        bounds = self._valid_media_bounds(processed_inputs)
        if bounds and len(bounds) != len(mappings):
            raise ValueError(
                "VideoNSA image_bound rows do not match placeholder ranges: "
                f"{len(bounds)} != {len(mappings)}"
            )

        features = []
        for data_index, mapping in enumerate(mappings):
            start = int(mapping["start_index"])
            end = int(mapping["end"])
            if bounds:
                bound_start, bound_end = bounds[data_index]
                if (bound_start, bound_end) != (start, end):
                    raise ValueError(
                        "VideoNSA token placeholder range does not match "
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
                    modality=str(mapping["modality"]),
                    identifier=str(mapping["identifier"]),
                    position=MMPlaceholderRange(offset=start, length=end - start),
                    parts=(part,),
                )
            )
        return features
