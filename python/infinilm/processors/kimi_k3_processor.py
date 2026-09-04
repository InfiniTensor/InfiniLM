import json
import os

import infinicore
import torch
from transformers import AutoTokenizer
from typing_extensions import override

from ..llm.scheduler import SchedulerOutput
from ..llm.static_scheduler import StaticSchedulerOutput
from .basic_llm_processor import BasicLLMProcessor
from .processor import register_processor


@register_processor("kimi_k3")
class KimiK3Processor(BasicLLMProcessor):
    def __init__(self, model_dir_path: str):
        with open(os.path.join(model_dir_path, "config.json"), "r") as file:
            config = json.load(file)
        self.media_token_id = int(config["media_placeholder_token_id"])
        dtype_name = config.get("dtype", "bfloat16")
        self.pixel_values_dtype = getattr(torch, str(dtype_name))
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            model_dir_path, trust_remote_code=True
        )
        self.tokenizer = getattr(
            self.processor,
            "tokenizer",
            AutoTokenizer.from_pretrained(model_dir_path, trust_remote_code=True),
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
        if videos or audios:
            raise NotImplementedError("Kimi K3 currently supports image input only")
        if not images:
            return self.tokenizer(
                prompt, return_tensors=return_tensors, add_special_tokens=False
            )

        medias = [{"type": "image", "image": image} for image in images]
        result = self.processor(
            medias=medias,
            text=prompt,
            return_tensors=return_tensors or "pt",
            **kwargs,
        )
        grids = result.pop("grid_thws")
        if grids.ndim == 1:
            grids = grids.unsqueeze(0)
        input_ids = result["input_ids"]
        source_ids = input_ids[0] if input_ids.ndim == 2 else input_ids
        expanded = []
        bounds = []
        image_idx = 0
        for token_id in source_ids.tolist():
            if int(token_id) != self.media_token_id:
                expanded.append(int(token_id))
                continue
            if image_idx >= grids.shape[0]:
                raise RuntimeError("Kimi K3 has more image placeholders than images")
            _, grid_h, grid_w = [int(value) for value in grids[image_idx].tolist()]
            if grid_h % 2 != 0 or grid_w % 2 != 0:
                raise RuntimeError("Kimi K3 image grid must be divisible by 2")
            image_tokens = (grid_h // 2) * (grid_w // 2)
            start = len(expanded)
            expanded.extend([self.media_token_id] * image_tokens)
            bounds.append([start, len(expanded)])
            image_idx += 1
        if image_idx != grids.shape[0]:
            raise RuntimeError("Kimi K3 has more images than image placeholders")

        result["input_ids"] = torch.tensor([expanded], dtype=source_ids.dtype)
        result["attention_mask"] = torch.ones_like(result["input_ids"])
        result["image_grid_thw"] = grids
        result["image_bound"] = torch.tensor(bounds, dtype=torch.int64)
        return result

    @override
    def apply_chat_template(self, conversation, **kwargs):
        return self.processor.apply_chat_template(conversation, **kwargs)

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
            scheduler_output, temperature, top_p, top_k, **kwargs
        )
        if isinstance(scheduler_output, SchedulerOutput):
            self._append_multimodal_inputs(model_inputs, scheduler_output)

        init_indices = []
        final_indices = []
        for request in scheduler_output.scheduled_requests:
            if request.mamba_cache_index is None:
                raise RuntimeError(
                    f"Request {request.request_id} has no mamba cache index"
                )
            init_indices.append(
                0 if scheduler_output.is_prefill else request.mamba_cache_index
            )
            final_indices.append(request.mamba_cache_index)
        model_inputs["mamba_init_state_indices"] = infinicore.from_list(
            init_indices, dtype=infinicore.int32
        )
        model_inputs["mamba_final_state_indices"] = infinicore.from_list(
            final_indices, dtype=infinicore.int32
        )
        return model_inputs

    def _append_multimodal_inputs(
        self, model_inputs: dict, scheduler_output: SchedulerOutput
    ) -> None:
        pixel_values = []
        grids = []
        bounds = []
        request_ids = []
        if not scheduler_output.is_prefill:
            return

        for request_id, request in enumerate(scheduler_output.scheduled_requests):
            processed = request.processed_inputs
            if processed is None or "pixel_values" not in processed:
                continue
            image_grids = torch.as_tensor(processed["image_grid_thw"])
            if image_grids.ndim == 1:
                image_grids = image_grids.unsqueeze(0)
            image_bounds = torch.as_tensor(processed["image_bound"], dtype=torch.int64)
            pixels = torch.as_tensor(processed["pixel_values"])
            patch_counts = [int(grid.prod().item()) for grid in image_grids]
            image_pixels = list(torch.split(pixels, patch_counts, dim=0))
            if len(image_pixels) != image_bounds.shape[0]:
                raise RuntimeError("Kimi K3 image tensor count mismatch")

            num_cached = request.num_local_cached_tokens
            for image_pixel, grid, bound in zip(
                image_pixels, image_grids, image_bounds
            ):
                if int(bound[1]) <= num_cached:
                    continue
                if int(bound[0]) < num_cached:
                    raise RuntimeError("Kimi K3 cannot partially cache an image span")
                pixel_values.append(
                    infinicore.from_torch(image_pixel.to(self.pixel_values_dtype))
                )
                grids.append(infinicore.from_torch(grid))
                bounds.append(infinicore.from_torch(bound - num_cached))
                request_ids.append(request_id)

        if pixel_values:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["image_grid_thw"] = grids
            model_inputs["image_bound"] = bounds
            model_inputs["image_req_ids"] = request_ids

    @override
    def get_mm_token_index_list(
        self,
        prompt_token_ids,
        image_ids=None,
        video_ids=None,
        audio_ids=None,
        **kwargs,
    ):
        image_ids = image_ids or []
        spans = []
        image_index = 0
        index = 0
        while index < len(prompt_token_ids):
            if prompt_token_ids[index] != self.media_token_id:
                index += 1
                continue
            start = index
            while (
                index < len(prompt_token_ids)
                and prompt_token_ids[index] == self.media_token_id
            ):
                index += 1
            if image_index >= len(image_ids):
                raise RuntimeError("Kimi K3 image token span count mismatch")
            spans.append(
                {
                    "start_index": start,
                    "end_index": index - 1,
                    "identifier": image_ids[image_index],
                }
            )
            image_index += 1
        if image_index != len(image_ids):
            raise RuntimeError("Kimi K3 image token span count mismatch")
        return spans
