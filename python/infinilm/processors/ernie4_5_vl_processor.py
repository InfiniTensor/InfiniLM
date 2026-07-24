import json
import os

from transformers import AutoProcessor, AutoTokenizer
from typing_extensions import override

from ..llm.scheduler import SchedulerOutput
from ..llm.static_scheduler import StaticSchedulerOutput
from .basic_llm_processor import BasicLLMProcessor
from .processor import register_processor

DEFAULT_VIDEO_NUM_FRAMES = 8


@register_processor("ernie4_5_moe_vl")
class Ernie45VLProcessor(BasicLLMProcessor):
    def __init__(self, model_dir_path: str):
        self.pixel_values_dtype = None
        self.im_patch_id = None
        self.image_rescale_factor = 1.0 / 255.0
        self.image_mean = [0.48145466, 0.4578275, 0.40821073]
        self.image_std = [0.26862954, 0.26130258, 0.27577711]
        self.patch_size = 14

        preprocessor_path = os.path.join(model_dir_path, "preprocessor_config.json")
        if os.path.exists(preprocessor_path):
            with open(preprocessor_path, "r") as f:
                preprocessor_json = json.load(f)
            self.image_rescale_factor = preprocessor_json.get(
                "rescale_factor", self.image_rescale_factor
            )
            self.image_mean = preprocessor_json.get("image_mean", self.image_mean)
            self.image_std = preprocessor_json.get("image_std", self.image_std)
            self.patch_size = int(preprocessor_json.get("patch_size", self.patch_size))

        config_path = os.path.join(model_dir_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_json = json.load(f)
            self.im_patch_id = config_json.get("im_patch_id")
            dtype_name = config_json.get("torch_dtype") or config_json.get("dtype")
            if dtype_name is not None:
                import torch

                self.pixel_values_dtype = getattr(torch, str(dtype_name))

        try:
            self.processor = AutoProcessor.from_pretrained(
                model_dir_path, trust_remote_code=True
            )
            self.tokenizer = self.processor.tokenizer
        except Exception:
            self.processor = None
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_dir_path, trust_remote_code=True
            )

        if self.im_patch_id is None:
            self.im_patch_id = self.tokenizer.convert_tokens_to_ids(
                "<|IMAGE_PLACEHOLDER|>"
            )
        self.im_patch_id = int(self.im_patch_id)

    def _normalize_videos(self, videos):
        if not videos:
            return videos

        normalized = []
        for video in videos:
            if isinstance(video, tuple):
                normalized.append(video)
                continue
            if isinstance(video, str):
                from .videonsa_processor import decode_video_frames

                num_frames = int(
                    os.getenv(
                        "INFINILM_ERNIE_VIDEO_NUM_FRAMES",
                        os.getenv(
                            "INFINILM_VIDEONSA_VIDEO_NUM_FRAMES",
                            DEFAULT_VIDEO_NUM_FRAMES,
                        ),
                    )
                )
                video = decode_video_frames(video, num_frames)
            if isinstance(video, list):
                if not video:
                    continue
                if hasattr(video[0], "convert"):
                    import numpy as np

                    video = np.stack(
                        [np.array(frame.convert("RGB")) for frame in video], axis=0
                    )
            normalized.append(video)
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
        if not images and not videos and not audios:
            return self.tokenizer(
                prompt, return_tensors=return_tensors, add_special_tokens=False
            )

        if audios:
            raise NotImplementedError("ERNIE 4.5 VL processor does not support audio")
        if self.processor is None:
            raise RuntimeError("ERNIE 4.5 VL multimodal processor is not available")

        videos = self._normalize_videos(videos)
        processed = self.processor(
            text=prompt,
            images=images,
            videos=videos,
            return_tensors=return_tensors or "pt",
            **kwargs,
        )
        self._append_image_bound(processed)
        return processed

    def _append_image_bound(self, processed_inputs: dict) -> None:
        if "input_ids" not in processed_inputs:
            return

        import torch

        input_ids = processed_inputs["input_ids"]
        if isinstance(input_ids, torch.Tensor):
            token_ids = (
                input_ids[0].tolist() if input_ids.ndim == 2 else input_ids.tolist()
            )
        else:
            token_ids = (
                input_ids[0]
                if input_ids and isinstance(input_ids[0], list)
                else input_ids
            )

        bounds = []
        i = 0
        while i < len(token_ids):
            if int(token_ids[i]) != self.im_patch_id:
                i += 1
                continue
            start = i
            while i < len(token_ids) and int(token_ids[i]) == self.im_patch_id:
                i += 1
            bounds.append([start, i])

        processed_inputs["image_bound"] = torch.tensor(bounds, dtype=torch.int64)

    @override
    def apply_chat_template(
        self,
        conversation,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        **kwargs,
    ):
        normalized_conversation = []
        for message in conversation:
            content = message["content"]
            if not isinstance(content, list):
                normalized_conversation.append(message)
                continue

            normalized_content = []
            for item in content:
                item_type = item.get("type")
                if item_type == "text":
                    normalized_content.append(
                        {"type": "text", "text": item.get("text", "")}
                    )
                elif item_type == "image_url":
                    normalized_content.append({"type": "image"})
                elif item_type == "video_url":
                    normalized_content.append({"type": "video"})
                else:
                    raise NotImplementedError(
                        f"Unsupported ERNIE 4.5 VL content type: {item_type}"
                    )

            normalized_conversation.append(
                {"role": message.get("role", "user"), "content": normalized_content}
            )

        return self.tokenizer.apply_chat_template(
            conversation=normalized_conversation,
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
            model_inputs = self._build_model_input_from_static_scheduler_output(
                scheduler_output, temperature, top_p, top_k
            )
            self._append_ernie_position_ids(model_inputs, scheduler_output)
        elif isinstance(scheduler_output, SchedulerOutput):
            model_inputs = self._build_model_input_from_batch_scheduler_output(
                scheduler_output, temperature, top_p, top_k
            )
            self._append_ernie_mm_inputs(model_inputs, scheduler_output)
            self._append_ernie_position_ids(model_inputs, scheduler_output)
        else:
            raise ValueError(
                "scheduler_output must be an instance of SchedulerOutput or StaticSchedulerOutput"
            )
        return model_inputs

    def _position_delta(self, req) -> int:
        return int(getattr(req, "mrope_position_delta", 0))

    def _update_position_delta(self, req) -> None:
        import torch

        processed_inputs = req.processed_inputs
        if processed_inputs is None or "position_ids" not in processed_inputs:
            req.mrope_position_delta = 0
            return

        pos = processed_inputs["position_ids"]
        pos = pos if isinstance(pos, torch.Tensor) else torch.as_tensor(pos)
        if pos.ndim == 3 and pos.shape[0] == 1:
            last_pos = int(pos[0, -1, 0].item())
        elif pos.ndim == 2:
            last_pos = int(pos[-1, 0].item())
        else:
            req.mrope_position_delta = 0
            return
        req.mrope_position_delta = last_pos + 1 - len(req.get_input_tokens())

    def _build_prefill_position_ids(self, req, num_cached: int, compute_len: int):
        import torch

        processed_inputs = req.processed_inputs
        if processed_inputs is None or "position_ids" not in processed_inputs:
            pos = torch.arange(num_cached, num_cached + compute_len, dtype=torch.int64)
            return pos

        pos = processed_inputs["position_ids"]
        pos = pos if isinstance(pos, torch.Tensor) else torch.as_tensor(pos)
        if pos.ndim == 3 and pos.shape[0] == 1:
            return pos[:, num_cached : num_cached + compute_len, :].contiguous()
        if pos.ndim == 2:
            return pos[num_cached : num_cached + compute_len, :].contiguous()
        raise RuntimeError("ERNIE 4.5 VL position_ids must have shape [1, seq, 3]")

    def _append_ernie_position_ids(self, model_inputs: dict, scheduler_output) -> None:
        import infinicore
        import torch

        position_chunks = []
        has_3d = False
        for req in scheduler_output.scheduled_requests:
            if scheduler_output.is_prefill:
                self._update_position_delta(req)
                num_cached = req.num_local_cached_tokens
                compute_len = len(req.get_input_tokens()) - num_cached
                pos = self._build_prefill_position_ids(req, num_cached, compute_len)
            else:
                current_position = (
                    req.get_total_length() - 1 + self._position_delta(req)
                )
                pos = torch.tensor([current_position], dtype=torch.int64)
            has_3d = has_3d or pos.ndim == 3
            position_chunks.append(pos)

        if not position_chunks:
            return
        if has_3d:
            normalized = []
            for pos in position_chunks:
                if pos.ndim == 1:
                    pos = pos.view(1, -1, 1).expand(1, -1, 3)
                elif pos.ndim == 2:
                    pos = pos.view(1, pos.shape[0], pos.shape[1])
                normalized.append(pos)
            model_inputs["position_ids"] = infinicore.from_torch(
                torch.cat(normalized, dim=1).contiguous()
            )
        else:
            model_inputs["position_ids"] = infinicore.from_torch(
                torch.cat(
                    [pos.flatten() for pos in position_chunks], dim=0
                ).contiguous()
            )

    def _normalize_image_rows(self, image_rows):
        import torch

        image_rows = image_rows.to(torch.float32) * float(self.image_rescale_factor)
        repeat = self.patch_size * self.patch_size
        mean = torch.tensor(
            self.image_mean, dtype=torch.float32, device=image_rows.device
        )
        std = torch.tensor(
            self.image_std, dtype=torch.float32, device=image_rows.device
        )
        mean = mean.repeat_interleave(repeat).view(1, -1)
        std = std.repeat_interleave(repeat).view(1, -1)
        return (image_rows - mean) / std

    def _append_ernie_mm_inputs(
        self, model_inputs: dict, scheduler_output: SchedulerOutput
    ) -> None:
        import infinicore
        import torch

        pixel_values = []
        image_bound = []
        grid_thw = []
        image_req_ids = []

        for req_id, req in enumerate(scheduler_output.scheduled_requests):
            processed_inputs = req.processed_inputs
            if (
                not scheduler_output.is_prefill
                or processed_inputs is None
                or "images" not in processed_inputs
            ):
                continue

            bounds = processed_inputs.get("image_bound")
            grids = processed_inputs.get("grid_thw")
            if bounds is None or grids is None:
                raise RuntimeError(
                    "ERNIE 4.5 VL multimodal input requires image_bound and grid_thw"
                )

            images = processed_inputs["images"]
            images = (
                images if isinstance(images, torch.Tensor) else torch.as_tensor(images)
            )
            grids = grids if isinstance(grids, torch.Tensor) else torch.as_tensor(grids)
            bounds = (
                bounds if isinstance(bounds, torch.Tensor) else torch.as_tensor(bounds)
            )
            if bounds.ndim == 3 and bounds.shape[0] == 1:
                bounds = bounds.squeeze(0)
            if grids.ndim == 1:
                grids = grids.view(1, 3)
            if bounds.ndim != 2 or bounds.shape[-1] != 2:
                raise RuntimeError(
                    "ERNIE 4.5 VL image_bound must have shape [num_media, 2]"
                )
            if grids.ndim != 2 or grids.shape[-1] != 3:
                raise RuntimeError(
                    "ERNIE 4.5 VL grid_thw must have shape [num_media, 3]"
                )
            if bounds.shape[0] != grids.shape[0]:
                raise RuntimeError("ERNIE 4.5 VL media bounds and grids count mismatch")

            num_cached = req.num_local_cached_tokens
            partial_cached = (bounds[:, 0] < num_cached) & (bounds[:, 1] > num_cached)
            if partial_cached.any().item():
                raise RuntimeError(
                    "ERNIE 4.5 VL does not support partially cached multimodal spans"
                )

            row_offset = 0
            for image_idx in range(bounds.shape[0]):
                grid = grids[image_idx].to(torch.int64)
                rows = int(grid[0].item() * grid[1].item() * grid[2].item())
                image_rows = images[row_offset : row_offset + rows]
                row_offset += rows
                if bounds[image_idx, 1].item() <= num_cached:
                    continue
                image_rows = self._normalize_image_rows(image_rows)
                if self.pixel_values_dtype is not None:
                    image_rows = image_rows.to(self.pixel_values_dtype)
                pixel_values.append(image_rows.contiguous())
                image_bound.append((bounds[image_idx] - num_cached).to(torch.int64))
                grid_thw.append(grid.contiguous())
                image_req_ids.append(req_id)

        if pixel_values:
            model_inputs["pixel_values"] = [
                infinicore.from_torch(t) for t in pixel_values
            ]
            model_inputs["image_bound"] = [
                infinicore.from_torch(t) for t in image_bound
            ]
            model_inputs["tgt_sizes"] = [infinicore.from_torch(t) for t in grid_thw]
            model_inputs["image_req_ids"] = image_req_ids

    @override
    def get_mm_token_index_list(
        self, prompt_token_ids, image_ids=None, video_ids=None, audio_ids=None, **kwargs
    ):
        mm_token_index_list = []
        image_ids = image_ids or []
        video_ids = video_ids or []
        if image_ids and video_ids:
            raise NotImplementedError(
                "ERNIE 4.5 VL cache mapping does not support mixed image and video inputs yet"
            )
        media_ids = image_ids if image_ids else video_ids

        media_idx = 0
        i = 0
        while i < len(prompt_token_ids):
            if int(prompt_token_ids[i]) != self.im_patch_id:
                i += 1
                continue

            start = i
            while (
                i < len(prompt_token_ids)
                and int(prompt_token_ids[i]) == self.im_patch_id
            ):
                i += 1

            if media_idx >= len(media_ids):
                raise RuntimeError(
                    "The number of ERNIE 4.5 VL multimodal token spans exceeds media inputs"
                )
            mm_token_index_list.append(
                {
                    "start_index": start,
                    "end_index": i - 1,
                    "identifier": media_ids[media_idx],
                }
            )
            media_idx += 1

        if media_idx != len(media_ids):
            raise RuntimeError(
                "The number of ERNIE 4.5 VL multimodal token spans does not match media inputs"
            )
        return mm_token_index_list
