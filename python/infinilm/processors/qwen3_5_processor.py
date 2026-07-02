from typing_extensions import override

from transformers import AutoProcessor, AutoTokenizer

from ..llm.scheduler import SchedulerOutput
from ..llm.static_scheduler import StaticSchedulerOutput
from .basic_llm_processor import BasicLLMProcessor
from .processor import register_processor


@register_processor("qwen3_5")
class Qwen35Processor(BasicLLMProcessor):
    def __init__(self, model_dir_path: str):
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

        return self.processor(
            text=prompt,
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
                    normalized_content.append({"type": "text", "text": item.get("text", "")})
                elif item_type == "image_url":
                    normalized_content.append({"type": "image"})
                elif item_type == "video_url":
                    normalized_content.append({"type": "video"})
                else:
                    raise NotImplementedError(f"Unsupported Qwen3.5 content type: {item_type}")

            normalized_conversation.append(
                {"role": message.get("role", "user"), "content": normalized_content}
            )

        template_owner = self.processor if self.processor is not None else self.tokenizer
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
        elif isinstance(scheduler_output, SchedulerOutput):
            model_inputs = self._build_model_input_from_batch_scheduler_output(
                scheduler_output, temperature, top_p, top_k
            )
            self._append_qwen35_mm_inputs(model_inputs, scheduler_output)
        else:
            raise ValueError(
                "scheduler_output must be an instance of SchedulerOutput or StaticSchedulerOutput"
            )

        # TODO(qwen3_5): The scheduler should own stable mamba cache ids. For now
        # use a per-forward arange so the C++ model input and mamba metadata path
        # can be exercised without encoding cache policy in the processor.
        num_requests = len(scheduler_output.scheduled_requests)
        init_indices = list(range(num_requests))
        final_indices = list(range(num_requests))

        import infinicore

        model_inputs["mamba_init_state_indices"] = infinicore.from_list(
            init_indices, dtype=infinicore.int32
        )
        model_inputs["mamba_final_state_indices"] = infinicore.from_list(
            final_indices, dtype=infinicore.int32
        )
        return model_inputs

    def _append_qwen35_mm_inputs(
        self, model_inputs: dict, scheduler_output: SchedulerOutput
    ) -> None:
        import infinicore
        import torch

        pixel_values = []
        image_req_ids = []
        for req_id, req in enumerate(scheduler_output.scheduled_requests):
            processed_inputs = req.processed_inputs
            if (
                not scheduler_output.is_prefill
                or processed_inputs is None
                or "pixel_values" not in processed_inputs
            ):
                continue

            pixel_value = processed_inputs["pixel_values"]
            if isinstance(pixel_value, list):
                pixel_values.extend(pixel_value)
            else:
                pixel_values.append(pixel_value)
            image_req_ids.append(req_id)

        if pixel_values:
            pixel_values = [
                infinicore.from_torch(t if isinstance(t, torch.Tensor) else torch.as_tensor(t))
                for t in pixel_values
            ]
            model_inputs["pixel_values"] = pixel_values
            model_inputs["image_req_ids"] = image_req_ids

    @override
    def get_mm_token_index_list(
        self, prompt_token_ids, image_ids=None, video_ids=None, audio_ids=None, **kwargs
    ):
        mm_token_index_list = []
        

        return mm_token_index_list
