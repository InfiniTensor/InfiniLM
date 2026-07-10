import json
import os
import time

from .base import BaseBenchmark


class TransformersBenchmark(BaseBenchmark):
    """Hugging Face Transformers backend."""

    def __init__(
        self,
        model_dir_path,
        device_type_str="cpu",
        tensor_parallel_size=1,
        benchmark="ceval",
    ):
        import torch
        import transformers

        super().__init__(benchmark)

        supported_devices = {"cpu", "cuda", "mlu", "musa", "npu"}
        if device_type_str not in supported_devices:
            raise ValueError(
                f"Transformers backend unsupported device type: {device_type_str}"
            )
        self.device = torch.device(device_type_str)
        self.tensor_parallel_size = tensor_parallel_size

        with open(os.path.join(model_dir_path, "config.json"), "r") as f:
            self.config_dict = json.load(f)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_dir_path, trust_remote_code=True
        )

        print("Loading model with Transformers backend...")
        load_kwargs = {
            "dtype": "auto",
            "trust_remote_code": True,
        }
        if tensor_parallel_size > 1:
            if device_type_str != "cuda":
                raise ValueError(
                    "Transformers multi-GPU evaluation requires a CUDA device. "
                    f"Got device_type_str={device_type_str!r}."
                )
            available_devices = torch.cuda.device_count()
            if available_devices < tensor_parallel_size:
                raise ValueError(
                    f"Requested tp={tensor_parallel_size}, but only "
                    f"{available_devices} CUDA devices are visible."
                )
            load_kwargs["device_map"] = "auto"
            print(
                "Transformers multi-GPU device_map=auto enabled "
                f"for {tensor_parallel_size} visible CUDA devices"
            )

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_dir_path,
            **load_kwargs,
        )
        if tensor_parallel_size <= 1:
            self.model = self.model.to(self.device)
        self.model.eval()
        self.input_device = self.model.get_input_embeddings().weight.device
        print("Transformers model loaded successfully")

        eos_token_id = self.config_dict.get("eos_token_id")
        self.eos_token_id = (
            [eos_token_id] if isinstance(eos_token_id, int) else eos_token_id
        )

    def _synchronize(self):
        import torch

        device_module = getattr(torch, self.device.type, None)
        synchronize = getattr(device_module, "synchronize", None)
        if synchronize is not None:
            synchronize()

    def generate(self, *args, max_steps=500, topp_=1.0, topk_=1, temperature_=1.0):
        import torch

        prompt = self.render_input_content(*args)
        print(prompt, end="", flush=True)
        tokens = self.encode_text(prompt)
        input_ids = torch.tensor([tokens], device=self.input_device)

        self._synchronize()
        start_time = time.perf_counter()
        outputs = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_steps,
            do_sample=temperature_ > 0,
            temperature=temperature_,
            top_k=topk_,
            top_p=topp_,
            eos_token_id=self.eos_token_id,
            pad_token_id=(
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            ),
        )
        self._synchronize()

        generated_ids = outputs[0][len(tokens) :]
        output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return self.record_generation(
            output_text,
            len(tokens),
            generated_ids.numel(),
            start_time,
        )

    def destroy_model_instance(self):
        del self.model
        print("Transformers model destroyed")
