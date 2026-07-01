import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../python"))

from infinilm.base_config import BaseConfig
from infinilm.llm.llm import LLM
from infinilm.llm.sampling_params import SamplingParams
from infinilm.processors import AutoInfinilmProcessor
from infinilm.processors.videonsa_processor import decode_video_frames


VIDEO_AUTO_MIN_FRAMES = 4
VIDEO_AUTO_MAX_FRAMES = 8
VIDEO_AUTO_SAMPLE_FPS = 1.0
VIDEO_AUTO_MAX_PIXELS_CAP = 50176


def as_int_list(value):
    if isinstance(value, list):
        return [int(item) for item in value]
    return [int(value)]


def is_cli_arg_set(name):
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv[1:])


def probe_video_metadata(video_path):
    try:
        import cv2
    except Exception:
        cv2 = None

    if cv2 is not None:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            try:
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            finally:
                cap.release()
            duration = frame_count / fps if frame_count > 0 and fps > 0 else 0.0
            return {
                "frame_count": frame_count,
                "fps": fps,
                "width": width,
                "height": height,
                "duration": duration,
                "source": "cv2",
            }

    try:
        from torchvision.io import read_video_timestamps

        pts, fps = read_video_timestamps(video_path, pts_unit="sec")
    except Exception:
        return {}

    frame_count = len(pts)
    duration = float(pts[-1]) if frame_count > 0 else 0.0
    return {
        "frame_count": frame_count,
        "fps": float(fps or 0.0),
        "width": 0,
        "height": 0,
        "duration": duration,
        "source": "torchvision_timestamps",
    }


def apply_video_auto_args(cfg):
    if not cfg.video:
        return {}
    meta = probe_video_metadata(cfg.video)
    frame_count = meta.get("frame_count", 0)
    duration = meta.get("duration", 0.0)
    width = meta.get("width", 0)
    height = meta.get("height", 0)

    if cfg.video_num_frames is None:
        if duration > 0:
            inferred = int(round(duration * VIDEO_AUTO_SAMPLE_FPS))
        elif frame_count > 0:
            inferred = frame_count
        else:
            inferred = VIDEO_AUTO_MIN_FRAMES
        if frame_count > 0:
            inferred = min(inferred, frame_count)
        cfg.video_num_frames = max(
            VIDEO_AUTO_MIN_FRAMES,
            min(VIDEO_AUTO_MAX_FRAMES, inferred),
        )

    if cfg.video_max_pixels is None:
        source_pixels = (
            width * height if width > 0 and height > 0 else VIDEO_AUTO_MAX_PIXELS_CAP
        )
        cfg.video_max_pixels = min(source_pixels, VIDEO_AUTO_MAX_PIXELS_CAP)

    return meta


def apply_multimodal_env(cfg):
    if cfg.image and not cfg.video and cfg.image_max_pixels is not None:
        os.environ["INFINILM_VIDEONSA_IMAGE_MAX_PIXELS"] = str(cfg.image_max_pixels)
    if cfg.image and not cfg.video and cfg.image_min_pixels is not None:
        os.environ["INFINILM_VIDEONSA_IMAGE_MIN_PIXELS"] = str(cfg.image_min_pixels)
    if cfg.video_num_frames is not None:
        os.environ["INFINILM_VIDEONSA_VIDEO_NUM_FRAMES"] = str(cfg.video_num_frames)
    if cfg.video_max_pixels is not None:
        os.environ["INFINILM_VIDEONSA_VIDEO_MAX_PIXELS"] = str(cfg.video_max_pixels)
    if cfg.video_min_pixels is not None:
        os.environ["INFINILM_VIDEONSA_VIDEO_MIN_PIXELS"] = str(cfg.video_min_pixels)


def normalize_bench_defaults(cfg):
    if cfg.video and not is_cli_arg_set("--video-max-pixels"):
        cfg.video_max_pixels = None
    if not cfg.image and not cfg.video:
        cfg.image = "/data-aisoft/pepe/images/bus.jpg"
    if cfg.prompt == "How are you":
        cfg.prompt = "describe the video" if cfg.video else "describe the image"


def make_prompt(tokenizer, target_len, prompt_seed):
    seed = prompt_seed.rstrip() + " "
    seed_ids = tokenizer.encode(seed)
    if not seed_ids:
        raise RuntimeError("Tokenizer returned no tokens for the benchmark seed prompt")
    repeat = (target_len + len(seed_ids) - 1) // len(seed_ids)
    return tokenizer.decode((seed_ids * repeat)[:target_len], skip_special_tokens=True)


def make_messages(prompt, image_path, video_path, video_payload, batch_size):
    content = [{"type": "text", "text": prompt}]
    if video_path:
        content = [
            {"type": "video_url", "video_url": {"url": video_payload or video_path}}
        ] + content
    elif image_path:
        content = [{"type": "image_url", "image_url": {"url": image_path}}] + content
    return [[{"role": "user", "content": content}] for _ in range(batch_size)]


def run_case(model, tokenizer, cfg, video_payload, batch_size, input_len, output_len):
    prompt = make_prompt(tokenizer, input_len, cfg.prompt)
    messages = make_messages(prompt, cfg.image, cfg.video, video_payload, batch_size)
    sampling = SamplingParams(
        max_tokens=output_len,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        ignore_eos=cfg.ignore_eos,
    )

    if cfg.warmup:
        model.chat(messages=messages, sampling_params=sampling, use_tqdm=False)

    start = time.perf_counter()
    outputs = model.chat(messages=messages, sampling_params=sampling, use_tqdm=False)
    elapsed_ms = (time.perf_counter() - start) * 1000
    total_new_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    print(
        "case "
        f"batch_size={batch_size} input_len={input_len} output_len={output_len} "
        f"actual_output_tokens={total_new_tokens} "
        f"elapsed_ms={elapsed_ms:.2f} "
        f"output_tok_per_s={total_new_tokens / (elapsed_ms / 1000):.2f}"
    )
    if outputs and not cfg.skip_output:
        print("=== sample prompt ===")
        print(outputs[0].prompt)
        print("=== sample output ===")
        print(outputs[0].outputs[0].text)


def main():
    cfg = BaseConfig()
    normalize_bench_defaults(cfg)

    input_lens = as_int_list(cfg.input_len)
    output_lens = as_int_list(cfg.output_len)
    max_batch_size = max(int(cfg.batch_size), int(cfg.max_batch_size))
    max_cache_len = max(max(input_lens) + max(output_lens) + 4096, cfg.max_cache_len)
    cache_type = "paged" if cfg.enable_paged_attn else "static"

    video_meta = apply_video_auto_args(cfg)
    apply_multimodal_env(cfg)
    video_payload = decode_video_frames(cfg.video, cfg.video_num_frames)

    print(
        f"bench_config model={cfg.model} image={cfg.image} video={cfg.video} prompt={cfg.prompt!r} "
        f"device={cfg.device} paged={cfg.enable_paged_attn} attn={cfg.attn} "
        f"videonsa_nsa=always_on "
        f"image_max_pixels={cfg.image_max_pixels} "
        f"video_num_frames={cfg.video_num_frames} video_max_pixels={cfg.video_max_pixels} "
        f"video_predecoded={isinstance(video_payload, list)} "
        f"video_meta={video_meta}"
    )
    processor = AutoInfinilmProcessor.from_pretrained(cfg.model)
    tokenizer = processor.get_tokenizer()
    model = LLM(
        model_path=cfg.model,
        device=cfg.get_device_str(cfg.device),
        tensor_parallel_size=cfg.tp,
        cache_type=cache_type,
        max_batch_size=max_batch_size,
        max_tokens=max(output_lens),
        num_blocks=cfg.num_blocks,
        block_size=cfg.block_size,
        max_cache_len=max_cache_len,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        attn_backend=cfg.attn,
        enable_graph=cfg.enable_graph,
        weight_load_mode=cfg.weight_load_mode,
    )

    for input_len in input_lens:
        for output_len in output_lens:
            run_case(
                model,
                tokenizer,
                cfg,
                video_payload,
                cfg.batch_size,
                input_len,
                output_len,
            )


if __name__ == "__main__":
    main()
