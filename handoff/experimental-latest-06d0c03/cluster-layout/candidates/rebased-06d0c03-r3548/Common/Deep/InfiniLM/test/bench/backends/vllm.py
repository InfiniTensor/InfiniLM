import json
import os
import runpy
import sys
import time
from pathlib import Path

from .base import BaseBenchmark


class VLLMBenchmark(BaseBenchmark):
    """vLLM backend."""

    def __init__(
        self,
        model_dir_path,
        device_type_str="cuda",
        tensor_parallel_size=1,
        benchmark="ceval",
    ):
        import torch
        import transformers

        # Hygon exposes an NVML compatibility layer even for HIP builds. Ensure
        # both this process and vLLM model-inspection subprocesses select ROCm.
        if torch.version.hip is not None:
            compat_dir = Path(__file__).parent / "_vllm_hip_compat"
            os.environ["INFINILM_VLLM_FORCE_ROCM"] = "1"
            os.environ["PYTHONPATH"] = os.pathsep.join(
                [str(compat_dir), os.environ.get("PYTHONPATH", "")]
            ).rstrip(os.pathsep)
            sys.path.insert(0, str(compat_dir))
            runpy.run_path(str(compat_dir / "sitecustomize.py"))

        from vllm import LLM

        if device_type_str == "cpu":
            raise ValueError("vLLM backend does not support CPU device type.")

        super().__init__(benchmark)

        with open(os.path.join(model_dir_path, "config.json"), "r") as f:
            self.config_dict = json.load(f)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_dir_path, trust_remote_code=True
        )
        eos_token_id = self.config_dict.get("eos_token_id")
        self.eos_token_id = (
            [eos_token_id] if isinstance(eos_token_id, int) else eos_token_id
        )

        print("Loading model with vLLM backend...")
        self.llm = LLM(
            model=model_dir_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
        )
        print("vLLM model loaded successfully")

    def generate(self, *args, max_steps=500, topp_=1.0, topk_=1, temperature_=1.0):
        from vllm import SamplingParams

        prompt = self.render_input_content(*args)
        print(prompt, end="", flush=True)
        input_tokens = len(self.encode_text(prompt))
        sampling_params = SamplingParams(
            max_tokens=max_steps,
            temperature=temperature_,
            top_p=topp_,
            top_k=topk_,
            stop_token_ids=self.eos_token_id,
        )

        start_time = time.perf_counter()
        request_output = self.llm.generate(
            prompts=[prompt], sampling_params=sampling_params
        )[0]
        completion = request_output.outputs[0]
        return self.record_generation(
            completion.text,
            input_tokens,
            len(completion.token_ids),
            start_time,
        )

    def destroy_model_instance(self):
        del self.llm
        print("vLLM model destroyed")
