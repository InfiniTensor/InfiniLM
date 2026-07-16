import json
import os
import time

from .base import BaseBenchmark


class InfiniLMBenchmark(BaseBenchmark):
    """InfiniLM backend using the scheduler-backed high-level API."""

    def __init__(
        self,
        model_dir_path,
        device_type_str="cpu",
        tensor_parallel_size=1,
        benchmark="ceval",
        enable_paged_attn=False,
        enable_graph=False,
        attn_backend="default",
    ):
        from infinilm import LLM

        super().__init__(benchmark)

        device_map = {
            "cpu": "cpu",
            "nvidia": "cuda",
            "cambricon": "mlu",
            "ascend": "npu",
            "metax": "cuda",
            "moore": "musa",
            "iluvatar": "cuda",
            "hygon": "cuda",
            "cuda": "cuda",
            "mlu": "mlu",
            "musa": "musa",
            "npu": "npu",
        }
        try:
            device_name = device_map[device_type_str.lower()]
        except KeyError:
            supported = ", ".join(sorted(device_map))
            raise ValueError(
                f"unsupported device platform {device_type_str!r}; "
                f"expected one of: {supported}"
            ) from None

        with open(os.path.join(model_dir_path, "config.json"), "r") as f:
            self.config_dict = json.load(f)

        if enable_paged_attn and attn_backend == "default":
            attn_backend = "paged-attn"

        print("Loading model with InfiniLM backend...")
        print(f"Graph compilation: {'enabled' if enable_graph else 'disabled'}")
        print(f"Attention backend: {attn_backend}")

        self.model = LLM(
            model_path=model_dir_path,
            device=device_name,
            tensor_parallel_size=tensor_parallel_size,
            cache_type="paged" if enable_paged_attn else "static",
            max_batch_size=1,
            num_blocks=128,
            block_size=256,
            enable_graph=enable_graph,
            attn_backend=attn_backend,
        )
        self.processor = self.model.engine.processor
        self.tokenizer = self.processor.get_tokenizer()
        print("InfiniLM model loaded successfully")

    def generate(self, *args, max_steps=500, topp_=1.0, topk_=1, temperature_=1.0):
        from infinilm import SamplingParams

        prompt = self.render_input_content(*args)
        print(prompt, end="", flush=True)

        start_time = time.perf_counter()
        request_output = self.model.generate(
            prompts=prompt,
            sampling_params=SamplingParams(
                max_tokens=max_steps,
                temperature=temperature_,
                top_k=topk_,
                top_p=topp_,
            ),
            use_tqdm=False,
        )[0]

        completion = request_output.outputs[0]
        return self.record_generation(
            completion.text,
            len(request_output.prompt_token_ids or []),
            len(completion.token_ids),
            start_time,
        )

    def destroy_model_instance(self):
        del self.model
        print("InfiniLM model destroyed")
