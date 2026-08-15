import torch
from transformers import AutoConfig, AutoProcessor
from typing_extensions import override

from ..llm.scheduler import SchedulerOutput
from ..llm.static_scheduler import StaticSchedulerOutput
from .processor import register_processor
from .qwen3_5_processor import Qwen35Processor


@register_processor("qwen3_vl")
class Qwen3VLProcessor(Qwen35Processor):
    def __init__(self, model_dir_path: str):
        self.processor = AutoProcessor.from_pretrained(
            model_dir_path, trust_remote_code=True
        )
        self.tokenizer = self.processor.tokenizer
        self.config = AutoConfig.from_pretrained(model_dir_path, trust_remote_code=True)
        self.image_token_id = self.config.image_token_id
        vision_config = getattr(self.config, "vision_config", None)
        self.spatial_merge_size = int(getattr(vision_config, "spatial_merge_size", 2))
        text_config = getattr(self.config, "text_config", None)
        dtype_name = getattr(text_config, "dtype", None) or getattr(
            text_config, "torch_dtype", None
        )
        if isinstance(dtype_name, torch.dtype):
            self.pixel_values_dtype = dtype_name
        elif dtype_name is not None:
            self.pixel_values_dtype = getattr(torch, str(dtype_name))
        else:
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
        results = self.processor(**processor_kwargs)
        self._append_image_bound(results)
        return results

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

    def _append_static_mm_inputs(
        self, model_inputs: dict, req, num_cached: int
    ) -> None:
        processed_inputs = req.processed_inputs
        if processed_inputs is None or "pixel_values" not in processed_inputs:
            return

        import infinicore

        grid_thw = processed_inputs.get("image_grid_thw")
        bounds = processed_inputs.get("image_bound")
        if grid_thw is None or bounds is None:
            raise RuntimeError(
                "Qwen3-VL image input requires image_grid_thw and image_bound"
            )

        grids = self._as_grid_list(grid_thw)

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

        pixels = split_pixel_values(processed_inputs["pixel_values"], grids)
        bounds = bounds if isinstance(bounds, torch.Tensor) else torch.as_tensor(bounds)
        if bounds.ndim == 3 and bounds.shape[0] == 1:
            bounds = bounds.squeeze(0)
        if bounds.ndim != 2 or bounds.shape[-1] != 2:
            raise RuntimeError("Qwen3-VL image_bound must have shape [num_images, 2]")
        if len(pixels) != len(grids) or len(pixels) != bounds.shape[0]:
            raise RuntimeError("Qwen3-VL image input count mismatch")

        partial_cached = (bounds[:, 0] < num_cached) & (bounds[:, 1] > num_cached)
        if partial_cached.any().item():
            raise RuntimeError("Qwen3-VL does not support partially cached image spans")

        pixel_values = []
        image_grid_thw = []
        image_bound = []
        for image_idx, (pixel, grid) in enumerate(zip(pixels, grids)):
            if bounds[image_idx, 1].item() <= num_cached:
                continue
            pixel_values.append(
                infinicore.from_torch(
                    (
                        pixel
                        if isinstance(pixel, torch.Tensor)
                        else torch.as_tensor(pixel)
                    ).to(self.pixel_values_dtype)
                )
            )
            image_grid_thw.append(
                infinicore.from_torch(
                    grid if isinstance(grid, torch.Tensor) else torch.as_tensor(grid)
                )
            )
            image_bound.append(
                infinicore.from_torch((bounds[image_idx] - num_cached).to(torch.int64))
            )

        if pixel_values:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["image_grid_thw"] = image_grid_thw
            model_inputs["image_bound"] = image_bound
            model_inputs["image_req_ids"] = [0] * len(pixel_values)

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
            if scheduler_output.is_prefill:
                self._append_static_mm_inputs(
                    model_inputs,
                    scheduler_output.scheduled_requests[0],
                    scheduler_output.prefix_hit_len,
                )
        elif isinstance(scheduler_output, SchedulerOutput):
            model_inputs = self._build_model_input_from_batch_scheduler_output(
                scheduler_output, temperature, top_p, top_k
            )
            self._append_qwen35_mm_inputs(model_inputs, scheduler_output)
        else:
            raise ValueError(
                "scheduler_output must be an instance of SchedulerOutput or StaticSchedulerOutput"
            )

        return model_inputs

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
