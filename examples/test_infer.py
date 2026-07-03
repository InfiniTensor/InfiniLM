import os
import time

from infinilm.base_config import BaseConfig
from infinilm.llm.llm import LLM
from infinilm.moe_config import configure_moe_ep_backend
from infinilm.processors.videonsa_processor import decode_video_frames


def test(
    prompts: list[str],
    model_path,
    max_new_tokens=100,
    device="cpu",
    tp=1,
    moe_ep_backend="disabled",
    ep=1,
    enable_paged_attn=False,
    enable_graph=False,
    top_k=1,
    top_p=1.0,
    temperature=1.0,
    attn_backend="default",
    use_mla=False,
    image_path=None,
    video_path=None,
    video_num_frames=None,
    skip_load=False,
    weight_load_mode="async",
    skip_legacy_moe=False,
):
    model_path = os.path.expanduser(model_path)
    # ---------------------------------------------------------------------------- #
    #                        Create Model
    # ---------------------------------------------------------------------------- #
    if enable_paged_attn and attn_backend == "default":
        attn_backend = "paged-attn"

    model = LLM(
        model_path=model_path,
        device=device,
        tensor_parallel_size=tp,
        moe_ep_backend=moe_ep_backend,
        moe_ep_size=ep,
        cache_type="paged" if enable_paged_attn else "static",
        max_batch_size=len(prompts),
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        enable_graph=enable_graph,
        attn_backend=attn_backend,
        use_mla=use_mla,
        skip_load=skip_load,
        weight_load_mode=weight_load_mode,
        skip_legacy_moe=skip_legacy_moe,
    )

    conversations = [
        [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        for prompt in prompts
    ]
    if video_path is not None:
        video_payload = decode_video_frames(video_path, video_num_frames)
        for conversation in conversations:
            conversation[0]["content"] = [
                {"type": "video_url", "video_url": {"url": video_payload}}
            ] + conversation[0]["content"]
    elif image_path is not None:
        for conversation in conversations:
            conversation[0]["content"] = [
                {"type": "image_url", "image_url": {"url": image_path}}
            ] + conversation[0]["content"]

    t1 = time.time()
    print("=================== start generate ====================")

    outputs = model.chat(
        messages=conversations,
    )
    t2 = time.time()

    for i, output in enumerate(outputs):
        print(f"Resquest {i}:")
        print("===Query===")
        print(output.prompt)
        print("===Response===")
        print(output.outputs[0].text)
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

    if cfg.skip_legacy_moe:
        moe_ep_backend, ep = configure_moe_ep_backend(
            cfg.tp, cfg.dp, cfg.ep, cfg.moe_ep_backend, cfg.model
        )
    else:
        moe_ep_backend, ep = "disabled", 1

    test(
        prompts,
        model_path,
        max_new_tokens,
        device=device_str,
        tp=tp,
        moe_ep_backend=moe_ep_backend,
        ep=ep,
        enable_paged_attn=enable_paged_attn,
        enable_graph=enable_graph,
        top_k=cfg.top_k,
        top_p=cfg.top_p,
        temperature=cfg.temperature,
        attn_backend=cfg.attn,
        use_mla=cfg.use_mla,
        image_path=cfg.image,
        video_path=cfg.video,
        video_num_frames=cfg.video_num_frames,
        skip_load=cfg.skip_load,
        weight_load_mode=cfg.weight_load_mode,
        skip_legacy_moe=cfg.skip_legacy_moe,
    )
