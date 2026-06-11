import infinicore
from transformers import AutoTokenizer
from tokenizers import decoders as _dec
from infinilm.modeling_utils import get_model_state_dict
import infinilm
import argparse
import sys
import time
import os
from infinilm.base_config import BaseConfig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../python"))


def test(
    prompts: str | list[str],
    model_path,
    max_new_tokens=100,
    infini_device=infinicore.device("cpu", 0),
):
    model_path = os.path.expanduser(model_path)
    # ---------------------------------------------------------------------------- #
    #                        创建模型,
    # ---------------------------------------------------------------------------- #
    model = infinilm.AutoLlamaModel.from_pretrained(
        model_path,
        device=infini_device,
    )

    # ---------------------------------------------------------------------------- #
    #                        加载权重
    # ---------------------------------------------------------------------------- #
    model_param_infini = get_model_state_dict(
        model_path,
        device=infini_device,
        dtype=model.config.dtype,
    )

    model.load_state_dict(model_param_infini, strict=True)

    # ---------------------------------------------------------------------------- #
    #                        创建 tokenizer
    # ---------------------------------------------------------------------------- #
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if "llama" == model.model_type:
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
    else:
        raise ValueError(f"Unsupported model type: {model.model_type}")

    # ---------------------------------------------------------------------------- #
    #                        token编码
    # ---------------------------------------------------------------------------- #
    # prompt = "山东最高的山是？"
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
    input_ids_list = tokenizer.batch_encode_plus(input_contents)[
        "input_ids"
    ]  # List: [[1, 1128, 526, 366, 29892]]

    # ---------------------------------------------------------------------------- #
    #                        自回归生成
    # ---------------------------------------------------------------------------- #
    input_ids_infini = infinicore.from_list(input_ids_list, dtype=infinicore.int64)

    t1 = time.time()
    print("=================== start generate ====================")
    model.generate(
        input_ids_infini,
        max_new_tokens=max_new_tokens,
        tokenizer=tokenizer,
    )
    t2 = time.time()

    print(
        f"total_time: {round((t2 - t1) * 1000, 2)} ms",
    )


if __name__ == "__main__":
    cfg = BaseConfig()

    device_str = cfg.get_device_str(cfg.device)

    prompts = [cfg.prompt for _ in range(cfg.batch_size)]

    model_path = cfg.model
    max_new_tokens = cfg.max_new_tokens
    backend = cfg.backend

    if backend != "python":
        raise ValueError(f"Unsupported backend: {backend}.")

    infini_device = infinicore.device(device_str, 0)

    test(
        prompts,
        model_path,
        max_new_tokens,
        infini_device=infini_device,
    )
