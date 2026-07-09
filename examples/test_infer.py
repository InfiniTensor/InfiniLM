import time

from infinilm.base_config import BaseConfig
from infinilm.llm.llm import LLM
from infinilm.moe_config import configure_moe_ep_backend
from infinilm.processors.videonsa_processor import decode_video_frames


def test(
    prompts: list[str],
    config,
    image_path=None,
    video_path=None,
    video_num_frames=None,
):
    # ---------------------------------------------------------------------------- #
    #                        Create Model
    # ---------------------------------------------------------------------------- #
    model = LLM(config)

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

    prompts = [cfg.prompt for _ in range(cfg.batch_size)]

    if cfg.skip_legacy_moe:
        moe_ep_backend, ep = configure_moe_ep_backend(
            cfg.tp, cfg.dp, cfg.ep, cfg.moe_ep_backend, cfg.model
        )
    else:
        moe_ep_backend, ep = "disabled", 1

    cfg.moe_ep_backend = moe_ep_backend
    cfg.ep = ep
    cfg.max_batch_size = len(prompts)

    test(
        prompts,
        cfg,
        image_path=cfg.image,
        video_path=cfg.video,
        video_num_frames=cfg.video_num_frames,
    )
