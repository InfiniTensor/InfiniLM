import json
import os
import time

from .base import BaseBenchmark


def _fix_hf_meta_helper_tensors(model):
    """Materialize ERNIE remote-code helper tensors left on the meta device."""
    import torch

    patched = []
    vision_model = getattr(model, "vision_model", None) or getattr(
        model, "visual", None
    )
    rotary = (
        getattr(vision_model, "rotary_pos_emb", None)
        if vision_model is not None
        else None
    )
    inv_freq = getattr(rotary, "inv_freq", None) if rotary is not None else None
    if inv_freq is not None and getattr(inv_freq, "device", None).type == "meta":
        dim = int(inv_freq.numel() * 2)
        theta = float(getattr(rotary, "theta", 10000.0))
        rotary.inv_freq = 1.0 / theta ** (
            torch.arange(start=0, end=dim, step=2, dtype=torch.float32) / dim
        )
        patched.append("vision_rotary_inv_freq")

    for module in model.modules():
        experts_type_ids = getattr(module, "experts_type_ids", None)
        if (
            experts_type_ids is None
            or getattr(experts_type_ids, "device", None).type != "meta"
        ):
            continue
        config = getattr(module, "config", None)
        moe_num_experts = getattr(config, "moe_num_experts", None)
        if not isinstance(moe_num_experts, (list, tuple)):
            continue
        rebuilt = torch.zeros([sum(moe_num_experts)], dtype=torch.int64)
        offset = 0
        for idx, expert_num in enumerate(moe_num_experts):
            rebuilt[offset : offset + expert_num] = idx
            offset += expert_num
        module.experts_type_ids = rebuilt
        module.experts_type_mask = [
            module.experts_type_ids == idx for idx, _ in enumerate(moe_num_experts)
        ]
        patched.append("experts_type_ids")
    return patched


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

        self.use_processor_inputs = (
            self.config_dict.get("model_type") == "ernie4_5_moe_vl"
        )
        if self.use_processor_inputs:
            self.processor = transformers.AutoProcessor.from_pretrained(
                model_dir_path, trust_remote_code=True
            )
            self.tokenizer = self.processor.tokenizer
        else:
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
        patched_meta = _fix_hf_meta_helper_tensors(self.model)
        if patched_meta:
            print(f"Patched HF meta helper tensors: {patched_meta}")
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
        if self.use_processor_inputs:
            inputs = self.processor(
                prompt, return_tensors="pt", add_special_tokens=False
            )
            model_inputs = {
                key: value.to(self.input_device)
                for key, value in inputs.items()
                if value is not None
            }
            if "attention_mask" not in model_inputs:
                model_inputs["attention_mask"] = torch.ones_like(
                    model_inputs["input_ids"]
                )
            input_tokens = int(model_inputs["input_ids"].shape[-1])

            self._synchronize()
            start_time = time.perf_counter()
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=max_steps,
                do_sample=temperature_ > 0,
                temperature=temperature_,
                top_k=topk_,
                top_p=topp_,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id or 2,
                use_cache=False,
            )
            self._synchronize()

            generated_ids = outputs[0][input_tokens:]
            output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            return self.record_generation(
                output_text,
                input_tokens,
                generated_ids.numel(),
                start_time,
            )

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
