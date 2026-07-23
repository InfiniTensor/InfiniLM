import json
import os

from transformers import AutoProcessor, AutoTokenizer
from typing_extensions import override

from ..llm.scheduler import SchedulerOutput
from ..llm.static_scheduler import StaticSchedulerOutput
from .basic_llm_processor import BasicLLMProcessor
from .processor import register_processor


@register_processor("qwen3_5")
class Qwen35Processor(BasicLLMProcessor):
    supports_mixed_batch = False

    def __init__(self, model_dir_path: str):
        self.pixel_values_dtype = None
        config_path = os.path.join(model_dir_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_json = json.load(f)
            dtype_name = (
                config_json.get("torch_dtype")
                or config_json.get("dtype")
                or config_json.get("text_config", {}).get("torch_dtype")
                or config_json.get("text_config", {}).get("dtype")
            )
            self.spatial_merge_size = int(
                config_json.get("vision_config", {}).get("spatial_merge_size", 2)
            )
            if dtype_name is not None:
                import torch

                self.pixel_values_dtype = getattr(torch, str(dtype_name))
        else:
            self.spatial_merge_size = 2
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

        if self.processor is None:
            raise RuntimeError("Qwen3.5 multimodal processor is not available")

        media_kwargs = {}
        if images:
            media_kwargs["images"] = images
            images_kwargs = dict(kwargs.pop("images_kwargs", {}) or {})
            image_size = dict(images_kwargs.get("size", {}) or {})
            image_size.setdefault("shortest_edge", 3136)
            image_size.setdefault("longest_edge", 2000000)
            images_kwargs["size"] = image_size
            kwargs["images_kwargs"] = images_kwargs
        if videos:
            media_kwargs["videos"] = videos

        results = self.processor(
            text=prompt,
            return_tensors=return_tensors or "pt",
            **media_kwargs,
            **kwargs,
        )
        self._append_image_bound(results)
        return results

    def _get_image_token_id(self) -> int:
        image_token_id = getattr(self.tokenizer, "image_token_id", None)
        if image_token_id is None:
            image_token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        return int(image_token_id)

    def _append_image_bound(self, processed_inputs: dict) -> None:
        if (
            "input_ids" not in processed_inputs
            or "pixel_values" not in processed_inputs
        ):
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

        image_token_id = self._get_image_token_id()
        bounds = []
        i = 0
        while i < len(token_ids):
            if int(token_ids[i]) != image_token_id:
                i += 1
                continue
            start = i
            while i < len(token_ids) and int(token_ids[i]) == image_token_id:
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
                        f"Unsupported Qwen3.5 content type: {item_type}"
                    )

            normalized_conversation.append(
                {"role": message.get("role", "user"), "content": normalized_content}
            )

        template_owner = (
            self.processor if self.processor is not None else self.tokenizer
        )
        return template_owner.apply_chat_template(
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
            self._append_qwen35_position_ids(model_inputs, scheduler_output)
        elif isinstance(scheduler_output, SchedulerOutput):
            model_inputs = self._build_model_input_from_batch_scheduler_output(
                scheduler_output, temperature, top_p, top_k
            )
            self._append_qwen35_mm_inputs(model_inputs, scheduler_output)
            self._append_qwen35_position_ids(model_inputs, scheduler_output)
        else:
            raise ValueError(
                "scheduler_output must be an instance of SchedulerOutput or StaticSchedulerOutput"
            )

        init_indices = []
        final_indices = []
        for req in scheduler_output.scheduled_requests:
            if req.mamba_cache_index is None:
                raise RuntimeError(
                    f"Request {req.request_id} has no assigned mamba cache index"
                )
            if scheduler_output.is_prefill:
                init_indices.append(0)
            else:
                init_indices.append(req.mamba_cache_index)
            final_indices.append(req.mamba_cache_index)

        import infinicore

        model_inputs["mamba_init_state_indices"] = infinicore.from_list(
            init_indices, dtype=infinicore.int32
        )
        model_inputs["mamba_final_state_indices"] = infinicore.from_list(
            final_indices, dtype=infinicore.int32
        )
        return model_inputs

    def _request_uses_mrope(self, req) -> bool:
        if req.processed_inputs is None:
            return False
        if (
            "image_grid_thw" in req.processed_inputs
            and "image_bound" in req.processed_inputs
        ):
            return True
        return "video_grid_thw" in req.processed_inputs

    def _mrope_delta(self, req) -> int:
        return int(getattr(req, "mrope_position_delta", 0))

    def _as_grid_list(self, grid_thw):
        import torch

        if isinstance(grid_thw, list):
            return [
                g if isinstance(g, torch.Tensor) else torch.as_tensor(g)
                for g in grid_thw
            ]
        grid_tensor = (
            grid_thw
            if isinstance(grid_thw, torch.Tensor)
            else torch.as_tensor(grid_thw)
        )
        if grid_tensor.ndim == 1:
            return [grid_tensor]
        return [grid_tensor[i] for i in range(grid_tensor.shape[0])]

    def _get_image_bounds(self, processed_inputs):
        import torch

        bounds = processed_inputs.get("image_bound")
        if bounds is None:
            return None
        bounds = bounds if isinstance(bounds, torch.Tensor) else torch.as_tensor(bounds)
        if bounds.ndim == 3 and bounds.shape[0] == 1:
            bounds = bounds.squeeze(0)
        if bounds.ndim != 2 or bounds.shape[-1] != 2:
            raise RuntimeError("Qwen3.5 image_bound must have shape [num_images, 2]")
        return bounds.to(dtype=torch.int64)

    def _mrope_position_metadata(self, req):
        if not self._request_uses_mrope(req):
            return None
        processed_inputs = req.processed_inputs
        bounds = self._get_image_bounds(processed_inputs)
        if bounds is None or bounds.numel() == 0:
            return None
        grids = self._as_grid_list(processed_inputs["image_grid_thw"])
        if len(grids) != bounds.shape[0]:
            raise RuntimeError("Qwen3.5 image grid count does not match image bounds")
        return bounds, grids

    def _compute_mrope_delta(self, req) -> int:
        metadata = self._mrope_position_metadata(req)
        if metadata is None:
            return 0

        bounds, grids = metadata
        seq_len = len(req.get_input_tokens())
        cursor = 0
        current_pos = 0
        max_pos = -1
        for bound, grid in zip(bounds, grids):
            start = int(bound[0].item())
            end = int(bound[1].item())
            if start < cursor or end < start or end > seq_len:
                raise RuntimeError("Qwen3.5 image_bound is invalid")

            text_len = start - cursor
            if text_len > 0:
                max_pos = max(max_pos, current_pos + text_len - 1)
                current_pos += text_len

            grid = grid.cpu().flatten()
            grid_t = int(grid[0].item())
            grid_h = int(grid[1].item())
            grid_w = int(grid[2].item())
            if (
                grid_h % self.spatial_merge_size != 0
                or grid_w % self.spatial_merge_size != 0
            ):
                raise RuntimeError(
                    "Qwen3.5 image grid must be divisible by spatial_merge_size"
                )
            llm_grid_h = grid_h // self.spatial_merge_size
            llm_grid_w = grid_w // self.spatial_merge_size
            max_pos = max(
                max_pos, current_pos + max(grid_t, llm_grid_h, llm_grid_w) - 1
            )
            current_pos += max(llm_grid_h, llm_grid_w)
            cursor = end

        if cursor < seq_len:
            text_len = seq_len - cursor
            max_pos = max(max_pos, current_pos + text_len - 1)

        return max_pos + 1 - seq_len

    def _build_mrope_position_ids(self, req, num_cached: int, compute_len: int):
        import torch

        metadata = self._mrope_position_metadata(req)
        if metadata is None:
            pos = torch.arange(num_cached, num_cached + compute_len, dtype=torch.int64)
            return pos.view(1, -1).expand(3, -1)

        bounds, grids = metadata
        seq_len = len(req.get_input_tokens())
        full_pos = torch.empty((3, seq_len), dtype=torch.int64)
        cursor = 0
        current_pos = 0
        for bound, grid in zip(bounds, grids):
            start = int(bound[0].item())
            end = int(bound[1].item())
            if start > cursor:
                text_len = start - cursor
                text_pos = torch.arange(
                    current_pos, current_pos + text_len, dtype=torch.int64
                )
                full_pos[:, cursor:start] = text_pos.view(1, -1).expand(3, -1)
                current_pos += text_len

            grid = grid.cpu().flatten()
            grid_t = int(grid[0].item())
            grid_h = int(grid[1].item())
            grid_w = int(grid[2].item())
            llm_grid_h = grid_h // self.spatial_merge_size
            llm_grid_w = grid_w // self.spatial_merge_size
            spatial_tokens = llm_grid_h * llm_grid_w
            temporal_pos = (
                torch.arange(grid_t, dtype=torch.int64).repeat_interleave(
                    spatial_tokens
                )
                + current_pos
            )
            height_pos = (
                torch.arange(llm_grid_h, dtype=torch.int64)
                .view(1, llm_grid_h, 1)
                .expand(grid_t, llm_grid_h, llm_grid_w)
                .reshape(-1)
                + current_pos
            )
            width_pos = (
                torch.arange(llm_grid_w, dtype=torch.int64)
                .view(1, 1, llm_grid_w)
                .expand(grid_t, llm_grid_h, llm_grid_w)
                .reshape(-1)
                + current_pos
            )
            if temporal_pos.numel() != end - start:
                raise RuntimeError("Qwen3.5 image token span does not match image grid")
            full_pos[:, start:end] = torch.stack(
                [temporal_pos, height_pos, width_pos], dim=0
            )
            current_pos += max(llm_grid_h, llm_grid_w)
            cursor = end

        if cursor < seq_len:
            text_len = seq_len - cursor
            text_pos = torch.arange(
                current_pos, current_pos + text_len, dtype=torch.int64
            )
            full_pos[:, cursor:] = text_pos.view(1, -1).expand(3, -1)

        return full_pos[:, num_cached : num_cached + compute_len]

    def _update_request_mrope_delta(self, req) -> None:
        req.mrope_position_delta = self._compute_mrope_delta(req)

    def _append_qwen35_position_ids(self, model_inputs: dict, scheduler_output) -> None:
        import infinicore
        import torch

        position_chunks = []
        for req in scheduler_output.scheduled_requests:
            if scheduler_output.is_prefill:
                self._update_request_mrope_delta(req)
                num_cached = req.num_local_cached_tokens
                compute_len = len(req.get_input_tokens()) - num_cached
                position_chunks.append(
                    self._build_mrope_position_ids(req, num_cached, compute_len)
                )
            else:
                current_position = req.get_total_length() - 1 + self._mrope_delta(req)
                pos = torch.tensor([current_position], dtype=torch.int64)
                position_chunks.append(pos.view(1, 1).expand(3, -1))

        if position_chunks:
            model_inputs["position_ids"] = infinicore.from_torch(
                torch.cat(position_chunks, dim=1).contiguous()
            )

    def _append_qwen35_mm_inputs(
        self, model_inputs: dict, scheduler_output: SchedulerOutput
    ) -> None:
        import infinicore
        import torch

        pixel_values = []
        image_grid_thw = []
        image_bound = []
        image_req_ids = []

        def as_grid_list(grid_thw):
            if isinstance(grid_thw, list):
                return [
                    g if isinstance(g, torch.Tensor) else torch.as_tensor(g)
                    for g in grid_thw
                ]
            grid_tensor = (
                grid_thw
                if isinstance(grid_thw, torch.Tensor)
                else torch.as_tensor(grid_thw)
            )
            if grid_tensor.ndim == 1:
                return [grid_tensor]
            return [grid_tensor[i] for i in range(grid_tensor.shape[0])]

        def split_pixel_values(pixel_value, grids):
            if isinstance(pixel_value, list):
                return [
                    p if isinstance(p, torch.Tensor) else torch.as_tensor(p)
                    for p in pixel_value
                ]
            pixel_tensor = (
                pixel_value
                if isinstance(pixel_value, torch.Tensor)
                else torch.as_tensor(pixel_value)
            )
            if len(grids) == 1:
                return [pixel_tensor]
            counts = [int(g[0].item() * g[1].item() * g[2].item()) for g in grids]
            return list(torch.split(pixel_tensor, counts, dim=0))

        for req_id, req in enumerate(scheduler_output.scheduled_requests):
            processed_inputs = req.processed_inputs
            if (
                not scheduler_output.is_prefill
                or processed_inputs is None
                or "pixel_values" not in processed_inputs
            ):
                continue

            grid_thw = processed_inputs.get("image_grid_thw")
            bounds = processed_inputs.get("image_bound")
            if grid_thw is None or bounds is None:
                raise RuntimeError(
                    "Qwen3.5 image input requires image_grid_thw and image_bound"
                )

            grids = as_grid_list(grid_thw)
            pixels = split_pixel_values(processed_inputs["pixel_values"], grids)
            bounds = (
                bounds if isinstance(bounds, torch.Tensor) else torch.as_tensor(bounds)
            )
            if bounds.ndim == 3 and bounds.shape[0] == 1:
                bounds = bounds.squeeze(0)
            if bounds.ndim != 2 or bounds.shape[-1] != 2:
                raise RuntimeError(
                    "Qwen3.5 image_bound must have shape [num_images, 2]"
                )
            if len(pixels) != len(grids) or len(pixels) != bounds.shape[0]:
                raise RuntimeError("Qwen3.5 image input count mismatch")

            num_cached = req.num_local_cached_tokens
            partial_cached = (bounds[:, 0] < num_cached) & (bounds[:, 1] > num_cached)
            if partial_cached.any().item():
                raise RuntimeError(
                    "Qwen3.5 does not support partially cached image spans"
                )

            for image_idx, (pixel, grid) in enumerate(zip(pixels, grids)):
                if bounds[image_idx, 1].item() <= num_cached:
                    continue
                pixel_values.append(pixel)
                image_grid_thw.append(grid)
                image_bound.append((bounds[image_idx] - num_cached).to(torch.int64))
                image_req_ids.append(req_id)

        if pixel_values:
            if len(pixel_values) != len(image_grid_thw) or len(pixel_values) != len(
                image_bound
            ):
                raise RuntimeError("Qwen3.5 multimodal input count mismatch")
            pixel_values = [
                infinicore.from_torch(
                    (t if isinstance(t, torch.Tensor) else torch.as_tensor(t)).to(
                        self.pixel_values_dtype
                    )
                    if self.pixel_values_dtype is not None
                    else (t if isinstance(t, torch.Tensor) else torch.as_tensor(t))
                )
                for t in pixel_values
            ]
            image_grid_thw = [
                infinicore.from_torch(
                    t if isinstance(t, torch.Tensor) else torch.as_tensor(t)
                )
                for t in image_grid_thw
            ]
            image_bound = [
                infinicore.from_torch(
                    t
                    if isinstance(t, torch.Tensor)
                    else torch.as_tensor(t, dtype=torch.int64)
                )
                for t in image_bound
            ]
            model_inputs["pixel_values"] = pixel_values
            model_inputs["image_grid_thw"] = image_grid_thw
            model_inputs["image_bound"] = image_bound
            model_inputs["image_req_ids"] = image_req_ids

    @override
    def get_mm_token_index_list(
        self, prompt_token_ids, image_ids=None, video_ids=None, audio_ids=None, **kwargs
    ):
        mm_token_index_list = []
        image_ids = image_ids or []
        image_token_id = self._get_image_token_id()

        image_idx = 0
        i = 0
        while i < len(prompt_token_ids):
            if prompt_token_ids[i] != image_token_id:
                i += 1
                continue

            start = i
            while i < len(prompt_token_ids) and prompt_token_ids[i] == image_token_id:
                i += 1

            if image_idx >= len(image_ids):
                raise RuntimeError(
                    "The number of Qwen3.5 image token spans exceeds image inputs"
                )
            mm_token_index_list.append(
                {
                    "start_index": start,
                    "end_index": i - 1,
                    "identifier": image_ids[image_idx],
                }
            )
            image_idx += 1

        if image_idx != len(image_ids):
            raise RuntimeError(
                "The number of Qwen3.5 image token spans does not match image inputs"
            )
        return mm_token_index_list
