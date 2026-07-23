import json
import os
import sys
import time

from .base import BaseBenchmark

# Select an isolated Transformers runtime before datasets imports site-packages.
_transformers_python_path = os.environ.get("TRANSFORMERS_PYTHON_PATH")
if _transformers_python_path:
    _transformers_python_path = os.path.abspath(_transformers_python_path)
    if _transformers_python_path not in sys.path:
        sys.path.insert(0, _transformers_python_path)


class TransformersBenchmark(BaseBenchmark):
    """Hugging Face Transformers backend."""

    _resident_models = {}

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

        local_files_only = os.environ.get("TRANSFORMERS_LOCAL_FILES_ONLY", "1") != "0"
        if local_files_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_dir_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )

        print(
            f"Transformers runtime: {transformers.__version__} ({transformers.__file__})"
        )

        self.processor = None
        if self.config_dict.get("model_type") == "deepseek_v4":
            from infinilm.processors.deepseek_v4_processor import DeepseekV4Processor

            self.processor = DeepseekV4Processor(model_dir_path, self.tokenizer)
            self.tokenizer = self.processor.tokenizer
            print("Using DeepseekV4Processor chat template")

        print("Loading model with Transformers backend...")
        dtype_name = os.environ.get("TRANSFORMERS_DTYPE", "bfloat16").lower()
        dtype = {
            "auto": "auto",
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }.get(dtype_name)
        if dtype is None:
            raise ValueError(f"Unsupported TRANSFORMERS_DTYPE={dtype_name!r}")
        load_kwargs = {
            "dtype": dtype,
            "trust_remote_code": True,
            "local_files_only": local_files_only,
            "use_safetensors": True,
            "low_cpu_mem_usage": True,
            "experts_implementation": os.environ.get(
                "TRANSFORMERS_EXPERTS_IMPLEMENTATION", "grouped_mm"
            ),
            # Avoid SDPA/flash-attn issues on some ROCm/Hygon stacks.
            "attn_implementation": os.environ.get(
                "TRANSFORMERS_ATTN_IMPLEMENTATION", "eager"
            ),
        }
        offload_state_dict = os.environ.get("TRANSFORMERS_OFFLOAD_STATE_DICT")
        if offload_state_dict is not None:
            load_kwargs["offload_state_dict"] = offload_state_dict != "0"
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
            load_kwargs["offload_folder"] = os.environ.get(
                "TRANSFORMERS_OFFLOAD_FOLDER",
                os.path.join(model_dir_path, ".offload"),
            )
            os.makedirs(load_kwargs["offload_folder"], exist_ok=True)
            print(
                "Transformers multi-GPU device_map=auto enabled "
                f"for {tensor_parallel_size} visible CUDA devices"
            )
            print(f"Offload folder: {load_kwargs['offload_folder']}")

        self._keep_model_resident = (
            os.environ.get("TRANSFORMERS_KEEP_MODEL_RESIDENT", "1") != "0"
        )
        self._cache_key = (
            os.path.realpath(model_dir_path),
            device_type_str,
            tensor_parallel_size,
            dtype_name,
            load_kwargs["experts_implementation"],
            load_kwargs["attn_implementation"],
            load_kwargs.get("offload_state_dict"),
        )
        cached = (
            self._resident_models.get(self._cache_key)
            if self._keep_model_resident
            else None
        )
        if cached is not None:
            self.model = cached["model"]
            self.input_device = cached["input_device"]
            print("Reusing resident Transformers model (load time: 0.00 s)")
        else:
            load_start = time.perf_counter()
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model_dir_path,
                **load_kwargs,
            )
            load_elapsed = time.perf_counter() - load_start
            if tensor_parallel_size <= 1:
                self.model = self.model.to(self.device)
            self.model.eval()
            self.input_device = self.model.get_input_embeddings().weight.device
            if self._keep_model_resident:
                self._resident_models[self._cache_key] = {
                    "model": self.model,
                    "input_device": self.input_device,
                }
            print(
                f"Transformers model loaded successfully in {load_elapsed:.2f} s"
            )

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
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([tokens], device=self.input_device)
        attention_mask = torch.ones_like(input_ids)

        self._synchronize()
        start_time = time.perf_counter()
        do_sample = temperature_ > 0 and topk_ != 1
        generation_kwargs = {
            "max_new_tokens": max_steps,
            "do_sample": do_sample,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            ),
        }
        if do_sample:
            generation_kwargs.update(
                temperature=temperature_,
                top_k=topk_,
                top_p=topp_,
            )

        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs,
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

    def generate_batch(
        self,
        batch_args,
        max_steps=500,
        topp_=1.0,
        topk_=1,
        temperature_=1.0,
    ):
        import torch

        if not batch_args:
            return []

        prompts = [self.render_input_content(*args) for args in batch_args]
        if os.environ.get("TRANSFORMERS_ALLOW_VARIABLE_LENGTH_BATCH", "1") != "1":
            length_groups = {}
            for index, prompt in enumerate(prompts):
                token_count = len(
                    self.tokenizer.encode(prompt, add_special_tokens=False)
                )
                length_groups.setdefault(token_count, []).append(index)
            if len(length_groups) > 1:
                print(
                    "DeepSeek-V4 variable-length batch split into "
                    f"{len(length_groups)} exact-length group(s)"
                )
                output_texts = [None] * len(batch_args)
                for indices in length_groups.values():
                    group_outputs = self.generate_batch(
                        [batch_args[index] for index in indices],
                        max_steps=max_steps,
                        topp_=topp_,
                        topk_=topk_,
                        temperature_=temperature_,
                    )
                    for index, output_text in zip(indices, group_outputs):
                        output_texts[index] = output_text
                return output_texts

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Batch generation requires a pad or EOS token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            pad_token_id = self.tokenizer.pad_token_id

        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            encoded = self.tokenizer(
                prompts,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            )
        finally:
            self.tokenizer.padding_side = original_padding_side

        input_ids = encoded["input_ids"].to(self.input_device)
        attention_mask = encoded["attention_mask"].to(self.input_device)
        input_token_counts = attention_mask.sum(dim=1).tolist()
        do_sample = temperature_ > 0 and topk_ != 1
        generation_kwargs = {
            "max_new_tokens": max_steps,
            "do_sample": do_sample,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": pad_token_id,
        }
        if do_sample:
            generation_kwargs.update(
                temperature=temperature_,
                top_k=topk_,
                top_p=topp_,
            )

        self._synchronize()
        start_time = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs,
            )
        self._synchronize()
        elapsed = time.perf_counter() - start_time

        generated_ids = outputs[:, input_ids.shape[1] :]
        output_texts = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        total_input_tokens = int(sum(input_token_counts))
        total_new_tokens = generated_ids.numel()
        total_tokens = total_input_tokens + total_new_tokens
        for index, output_text in enumerate(output_texts):
            print(f"[batch item {index}] {output_text}")
        print(f"Batch total time: {elapsed * 1000:.2f} ms")
        print(f"Batch input tokens: {total_input_tokens}")
        print(f"Batch new tokens: {total_new_tokens}")
        print(f"Batch throughput: {total_tokens / elapsed:.2f} tok/s")
        self.total_tokens += total_tokens
        self.total_time += elapsed
        return output_texts

    def destroy_model_instance(self):
        if self._keep_model_resident and self._cache_key in self._resident_models:
            print("Transformers model kept resident for reuse")
            return
        del self.model
        print("Transformers model destroyed")
