import time
import os
from infinilm.base_config import BaseConfig
from infinilm.llm.llm import LLM


def test(
    prompts: list[str],
    model_path,
    max_new_tokens=100,
    device="cpu",
    tp=1,
    dtype="float16",
    enable_paged_attn=False,
    enable_graph=False,
    num_blocks=512,
    block_size=256,
    top_k=1,
    top_p=1.0,
    temperature=1.0,
    attn_backend="default",
    image_path=None,
    skip_load=False,
):
    model_path = os.path.expanduser(model_path)
    # ---------------------------------------------------------------------------- #
    #                        Create Model
    # ---------------------------------------------------------------------------- #
    if enable_paged_attn and attn_backend == "default":
        attn_backend = "flash-attn"

    model = LLM(
        model_path=model_path,
        device=device,
        dtype=dtype,
        tensor_parallel_size=tp,
        cache_type="paged" if enable_paged_attn else "static",
        max_batch_size=len(prompts),
        max_tokens=max_new_tokens,
        num_blocks=num_blocks,
        block_size=block_size,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        enable_graph=enable_graph,
        attn_backend=attn_backend,
        skip_load=skip_load,
    )

    conversations = [
        [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        for prompt in prompts
    ]
    if image_path is not None:
        for conversation in conversations:
            conversation["content"] = [
                {"type": "image_url", "image_url": {"url": image_path}}
            ] + conversation["content"]

    t1 = time.time()
    print("=================== start generate ====================")

    outputs = model.chat(
        messages=conversations,
    )
    t2 = time.time()

    log_tokens = os.environ.get("INFINI_LOG_DECODE_TOKENS", "0") == "1"
    for i, output in enumerate(outputs):
        print(f"Resquest {i}:")
        print("===Query===")
        print(output.prompt)
        print("===Response===")
        print(output.outputs[0].text)
        if log_tokens:
            print(f"===TokenIds_{i}===")
            print(",".join(str(t) for t in output.outputs[0].token_ids))
        print("")

    print(
        f"total_time: {round((t2 - t1) * 1000, 2)} ms",
    )


if __name__ == "__main__":
    cfg = BaseConfig()

    device_str = cfg.get_device_str(cfg.device)

    prompts = [cfg.prompt for _ in range(cfg.batch_size)]

    model_path = cfg.model

    max_new_tokens = cfg.max_new_tokens

    tp = cfg.tp

    enable_paged_attn = cfg.enable_paged_attn

    enable_graph = cfg.enable_graph

    test(
        prompts,
        model_path,
        max_new_tokens,
        device=device_str,
        tp=tp,
        dtype=cfg.dtype,
        enable_paged_attn=enable_paged_attn,
        enable_graph=enable_graph,
        num_blocks=cfg.num_blocks,
        block_size=cfg.block_size,
        top_k=cfg.top_k,
        top_p=cfg.top_p,
        temperature=cfg.temperature,
        attn_backend=cfg.attn,
        image_path=cfg.image,
        skip_load=cfg.skip_load,
    )
