import json
from pathlib import Path

import torch
from transformers import AutoProcessor

from .basic_llm_processor import BasicLLMProcessor
from .processor import InfinilmProcessor, register_processor


_IMAGE_PLACEHOLDER = "<|IMAGE_START|><|image@placeholder|><|IMAGE_END|>"
_VIDEO_PLACEHOLDER = "<|VIDEO_START|><|video@placeholder|><|VIDEO_END|>"


@register_processor("ernie4_5_moe_vl")
@register_processor("ernie4_5_vl_moe")
class Ernie4_5VLProcessor(InfinilmProcessor):
    def __init__(self, model_dir_path: str):
        self.model_dir_path = model_dir_path
        self.processor = AutoProcessor.from_pretrained(
            model_dir_path, trust_remote_code=True
        )
        if hasattr(self.processor, "eval"):
            self.processor.eval()
        self.tokenizer = self.processor.tokenizer
        self.image_processor = self.processor.image_processor

        # Keep the initial VL path bounded while the vision tower runs in eager kernels.
        max_patches = 1024
        if hasattr(self.processor, "image_max_pixels"):
            self.processor.image_max_pixels = max_patches * 28 * 28

        with open(Path(model_dir_path) / "config.json", "r") as f:
            self.config = json.load(f)
        self.image_patch_id = int(self.config["im_patch_id"])

    def _ensure_media_placeholders(
        self, prompt: str, has_images: bool, has_videos: bool
    ) -> str:
        prefix = ""
        if has_images and "<|image@placeholder|>" not in prompt:
            prefix += _IMAGE_PLACEHOLDER
        if has_videos and "<|video@placeholder|>" not in prompt:
            prefix += _VIDEO_PLACEHOLDER
        if prefix == "":
            return prompt
        return f"User: {prefix}{prompt}\nAssistant:"

    def __call__(
        self,
        prompt,
        images=None,
        videos=None,
        audios=None,
        return_tensors: str = None,
        **kwargs,
    ) -> dict:
        if audios:
            raise NotImplementedError("ERNIE-4.5-VL audio input is not wired yet")

        if images or videos:
            prompt = self._ensure_media_placeholders(
                prompt, bool(images), bool(videos)
            )
            result = self.processor(
                text=[prompt],
                images=images,
                videos=videos,
                padding=True,
                return_tensors="pt",
                **kwargs,
            )
            result = dict(result)
            result["image_bound"] = self._build_image_bound(result["input_ids"])
            self._prepare_visual_outputs(result)
            return self._convert_return_tensors(result, return_tensors)

        if return_tensors is None:
            return self.tokenizer(prompt, add_special_tokens=False)
        if return_tensors == "pt":
            return self.tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False
            )
        if return_tensors == "infini":
            return self._convert_return_tensors(
                self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False),
                return_tensors,
            )
        return self.tokenizer(
            prompt, return_tensors=return_tensors, add_special_tokens=False
        )

    def _normalize_pixel_values(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(pixel_values):
            pixel_values = torch.tensor(pixel_values)
        image_mean = torch.tensor(
            self.image_processor.image_mean, dtype=torch.float32
        ).reshape(1, 3)
        image_std = torch.tensor(
            self.image_processor.image_std, dtype=torch.float32
        ).reshape(1, 3)
        patch_size = int(self.config["vision_config"]["patch_size"])
        patch_size_squared = patch_size * patch_size
        image_mean = image_mean.repeat_interleave(patch_size_squared, dim=-1)
        image_std = image_std.repeat_interleave(patch_size_squared, dim=-1)
        rescale_factor = float(self.image_processor.rescale_factor)
        pixel_values = (
            rescale_factor * pixel_values.to(torch.float32) - image_mean
        ) / image_std
        return pixel_values.to(torch.bfloat16).contiguous()

    def _prepare_visual_outputs(self, result: dict) -> None:
        pixel_values = self._pop_visual_tensor(
            result, ["images", "pixel_values", "pixel_values_videos"]
        )
        if pixel_values is not None:
            result["pixel_values"] = self._normalize_pixel_values(pixel_values)

        grid_thw = self._pop_visual_tensor(
            result, ["grid_thw", "image_grid_thw", "video_grid_thw"]
        )
        if grid_thw is not None:
            result["tgt_sizes"] = grid_thw.to(torch.int64).contiguous()

    @staticmethod
    def _pop_visual_tensor(result: dict, keys: list[str]):
        tensors = []
        for key in keys:
            value = result.pop(key, None)
            if value is None:
                continue
            if not torch.is_tensor(value):
                value = torch.tensor(value)
            tensors.append(value)
        if not tensors:
            return None
        if len(tensors) == 1:
            return tensors[0]
        return torch.cat(tensors, dim=0)

    def _build_image_bound(self, input_ids: torch.Tensor) -> torch.Tensor:
        ids = input_ids[0].tolist()
        ranges = []
        start = None
        for idx, token_id in enumerate(ids):
            if token_id == self.image_patch_id:
                if start is None:
                    start = idx
            elif start is not None:
                ranges.append([start, idx])
                start = None
        if start is not None:
            ranges.append([start, len(ids)])
        if not ranges:
            raise ValueError("ERNIE VL input did not contain visual patch tokens")
        return torch.tensor([ranges], dtype=torch.int64)

    def _convert_return_tensors(self, result: dict, return_tensors: str):
        if return_tensors != "infini":
            return result
        import infinicore

        converted = {}
        for key, value in result.items():
            if isinstance(value, torch.Tensor):
                converted[key] = infinicore.from_torch(value.contiguous())
            else:
                converted[key] = value
        return converted

    def apply_chat_template(
        self,
        conversation,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        **kwargs,
    ):
        return self.tokenizer.apply_chat_template(
            conversation=conversation,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
            **kwargs,
        )

    def build_model_inputs(
        self,
        scheduler_output,
        temperature: float = 1.0,
        top_p: float = 0.8,
        top_k: int = 1,
        **kwargs,
    ) -> dict:
        helper = BasicLLMProcessor.__new__(BasicLLMProcessor)
        helper.tokenizer = self.tokenizer
        model_inputs = BasicLLMProcessor.build_model_inputs(
            helper, scheduler_output, temperature, top_p, top_k, **kwargs
        )
        if not scheduler_output.is_prefill:
            import infinicore
            import torch

            mrope_position_ids = []
            has_mrope_position_ids = False
            for req in scheduler_output.scheduled_requests:
                processed_inputs = req.processed_inputs
                if processed_inputs is not None and "position_ids" in processed_inputs:
                    pos = processed_inputs["position_ids"]
                    if pos.ndim == 3:
                        pos = pos[0]
                    decode_pos = int(pos.max().item()) + len(req.generated_token_ids)
                    mrope_position_ids.append(
                        torch.tensor([decode_pos, decode_pos, decode_pos], dtype=torch.int64)
                    )
                    has_mrope_position_ids = True
                else:
                    decode_pos = req.get_total_length() - 1
                    mrope_position_ids.append(
                        torch.tensor([decode_pos, decode_pos, decode_pos], dtype=torch.int64)
                    )
            if has_mrope_position_ids:
                model_inputs["position_ids"] = infinicore.from_torch(
                    torch.vstack(mrope_position_ids).contiguous()
                )
            return model_inputs

        import infinicore
        import torch

        pixel_values = []
        image_bound = []
        tgt_sizes = []
        image_req_ids = []
        mrope_position_ids = []
        has_mrope_position_ids = False
        for req_id, req in enumerate(scheduler_output.scheduled_requests):
            processed_inputs = req.processed_inputs
            num_cached = req.num_local_cached_tokens
            tokens_to_compute = req.get_input_tokens()[num_cached:]
            if processed_inputs is not None and "position_ids" in processed_inputs:
                pos = processed_inputs["position_ids"]
                if pos.ndim == 3:
                    pos = pos[0]
                mrope_position_ids.append(pos[num_cached : num_cached + len(tokens_to_compute)].to(torch.int64))
                has_mrope_position_ids = True
            else:
                mrope_position_ids.append(
                    torch.tensor(
                        [[pos, pos, pos] for pos in range(num_cached, num_cached + len(tokens_to_compute))],
                        dtype=torch.int64,
                    )
                )

            if processed_inputs is None or "pixel_values" not in processed_inputs:
                continue
            if num_cached != 0:
                raise NotImplementedError(
                    "ERNIE-4.5-VL prefix-cache reuse with image inputs is not wired yet"
                )
            pixel_values.append(
                infinicore.from_torch(processed_inputs["pixel_values"].contiguous())
            )
            image_bound.append(
                infinicore.from_torch(processed_inputs["image_bound"].contiguous())
            )
            tgt_sizes.append(
                infinicore.from_torch(processed_inputs["tgt_sizes"].contiguous())
            )
            image_req_ids.append(req_id)

        if pixel_values:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["image_bound"] = image_bound
            model_inputs["tgt_sizes"] = tgt_sizes
            model_inputs["image_req_ids"] = image_req_ids
        if has_mrope_position_ids:
            model_inputs["position_ids"] = infinicore.from_torch(
                torch.vstack(mrope_position_ids).contiguous()
            )
        return model_inputs

    def get_tokenizer(self):
        return self.tokenizer

    def get_mm_token_index_list(
        self, prompt_token_ids, image_ids=None, video_ids=None, audio_ids=None, **kwargs
    ):
        if audio_ids:
            raise NotImplementedError(
                "ERNIE-4.5-VL audio service input is not wired yet"
            )
        image_ids = image_ids or []
        video_ids = video_ids or []
        media_ids = list(image_ids) + list(video_ids)
        ranges = []
        start = None
        for idx, token_id in enumerate(prompt_token_ids):
            if token_id == self.image_patch_id:
                if start is None:
                    start = idx
            elif start is not None:
                ranges.append((start, idx))
                start = None
        if start is not None:
            ranges.append((start, len(prompt_token_ids)))

        if len(ranges) != len(media_ids):
            raise ValueError(
                "The number of ERNIE media patch ranges does not match the number of media inputs"
            )
        return [
            {
                "start_index": start,
                "end_index": end,
                "identifier": media_ids[idx],
            }
            for idx, (start, end) in enumerate(ranges)
        ]
