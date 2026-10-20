import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../python"))

from tokenizers import decoders as _dec
from infinilm import LLM, SamplingParams
from infinilm.base_config import BaseConfig


def test(
    prompts: str | list[str],
    model_path,
    max_new_tokens=100,
    device="cpu",
):
    model_path = os.path.expanduser(model_path)
    with open(os.path.join(model_path, "config.json"), "r") as f:
        hf_config = json.load(f)
    if hf_config.get("model_type") != "llama":
        raise ValueError(f"Unsupported model type: {hf_config.get('model_type')}")
    declared_dtype = hf_config.get("torch_dtype") or hf_config.get("dtype")
    dtype_kwargs = {}
    if declared_dtype is not None:
        if declared_dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError(f"Unsupported checkpoint dtype: {declared_dtype}")
        dtype_kwargs["dtype"] = declared_dtype

    llm = LLM(
        model_path=model_path,
        device=device,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        enable_graph=False,
        **dtype_kwargs,
    )
    try:
        tokenizer = llm.engine.tokenizer
        backend = getattr(tokenizer, "backend_tokenizer", None)
        target = getattr(backend, "_tokenizer", backend)
        norm = getattr(target, "normalizer", None)
        dec = getattr(target, "decoder", None)
        sn = repr(norm)[:800] if norm is not None else ""
        sd = repr(dec)[:800] if dec is not None else ""
        has_prepend = "Prepend" in sn
        has_strip = "Strip" in sd
        if has_prepend and has_strip:
            target.decoder = _dec.Sequence(
                [
                    _dec.Replace("▁", " "),
                    _dec.ByteFallback(),
                    _dec.Fuse(),
                ]
            )

        if isinstance(prompts, str):
            prompts = [prompts]
        input_contents = [
            tokenizer.apply_chat_template(
                conversation=[{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
            for prompt in prompts
        ]
        print(input_contents[0], end="", flush=True)

        t1 = time.time()
        print("=================== start generate ====================")
        outputs = llm.generate(
            prompts=input_contents,
            sampling_params=SamplingParams(
                top_k=1,
                top_p=0.8,
                temperature=1.0,
                max_tokens=max_new_tokens,
                ignore_eos=False,
            ),
            use_tqdm=False,
        )
        t2 = time.time()

        # The public batch API returns completed texts, not per-token stdout.
        for output in outputs:
            for completion in output.outputs:
                print(completion.text)
        print(f"total_time: {round((t2 - t1) * 1000, 2)} ms")
    finally:
        llm.close()


if __name__ == "__main__":
    cfg = BaseConfig()

    device_str = cfg.get_device_str(cfg.device)

    prompts = [cfg.prompt for _ in range(cfg.batch_size)]

    model_path = cfg.model
    max_new_tokens = cfg.max_new_tokens
    backend = cfg.backend

    if backend == "python":
        warnings.warn(
            "The python backend is deprecated for this example; using the native "
            "public LLM implementation. Choose cpp or infinilm instead.",
            FutureWarning,
            stacklevel=2,
        )
    elif backend not in ("cpp", "infinilm"):
        raise ValueError(f"Unsupported backend: {backend}.")

    test(
        prompts,
        model_path,
        max_new_tokens,
        device=device_str,
    )
