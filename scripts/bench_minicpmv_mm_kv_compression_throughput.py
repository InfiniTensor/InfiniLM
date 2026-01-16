import argparse
import json
import random
import time
from ctypes import POINTER, c_float, c_int, c_uint
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image

from libinfinicore_infer import DeviceType, KVCacheCStruct, KVCompressionConfigCStruct


def _build_minicpmv_prompt_and_inputs(
    *,
    model_dir: Path,
    llm,
    image_path: str,
    question: str,
    debug: bool,
) -> Tuple[torch.Tensor, List[int], torch.Tensor, torch.Tensor, dict]:
    # Build processor using the same tokenizer (mirrors scripts/minicpmv_chat.py).
    preproc_cfg = json.loads((model_dir / "preprocessor_config.json").read_text())
    from image_processing_minicpmv import MiniCPMVImageProcessor
    from processing_minicpmv import MiniCPMVProcessor

    image_processor = MiniCPMVImageProcessor(**preproc_cfg)
    processor = MiniCPMVProcessor(image_processor=image_processor, tokenizer=llm.tokenizer)

    # The vendored processor searches for the literal pattern `(<image>./</image>)`,
    # so we must include exactly one char + '/' inside the image tag.
    user_content = f"<image>./</image>\n{question}"
    prompt = llm.tokenizer.apply_chat_template(
        conversation=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_content},
        ],
        add_generation_prompt=True,
        tokenize=False,
    )

    img = Image.open(image_path).convert("RGB")
    batch = processor(text=prompt, images=[img], return_tensors="pt")

    input_ids = batch["input_ids"][0].to(dtype=torch.int64)
    attn = batch["attention_mask"][0].to(dtype=torch.bool)
    pad_left = int((~attn).sum().item())
    tokens = input_ids[pad_left:].to(dtype=torch.int32).contiguous()

    bounds_all = batch["image_bound"][0].to(dtype=torch.int64) - pad_left
    if debug and int(bounds_all.numel()) > 0:
        print("DEBUG pad_left:", pad_left)
        print("DEBUG tokens_len:", int(tokens.numel()))
        print("DEBUG bounds_all:", bounds_all.tolist())

    pixel_values_slices = batch["pixel_values"][0]
    tgt_sizes = batch["tgt_sizes"][0]

    # `image_bound` may include non-vision spans (e.g., <image_id>...</image_id>), which are not 64-token features.
    feature_len = int(preproc_cfg.get("image_feature_size", 64))
    kept = [b for b in bounds_all if int((b[1] - b[0]).item()) == feature_len]
    if len(kept) == 0:
        raise RuntimeError("No valid image_bound spans (feature_len=%d) found" % feature_len)
    bounds = torch.stack(kept, dim=0)

    if debug and bounds.shape[0] != bounds_all.shape[0]:
        print(
            f"INFO: filtered image_bound: total={bounds_all.shape[0]} feature_len={feature_len} kept={bounds.shape[0]}"
        )
        print("  image_bound_all (after left-pad adjust):", bounds_all.tolist())
        print("  image_bound_kept:", bounds.tolist())

    if len(pixel_values_slices) != int(bounds.shape[0]):
        if debug:
            print(f"WARNING: slice count mismatch: slices={len(pixel_values_slices)} bounds={bounds.shape[0]}")
        n = min(len(pixel_values_slices), int(bounds.shape[0]))
        bounds = bounds[:n]
        pixel_values_slices = pixel_values_slices[:n]
        tgt_sizes = tgt_sizes[:n]

    if len(pixel_values_slices) == 0:
        raise RuntimeError("No image slices to run vision.")

    # Return python list for slices to make iteration predictable.
    slices_list = [x for x in pixel_values_slices]
    return tokens, slices_list, tgt_sizes, bounds, preproc_cfg


def _compute_override_embeds(
    *,
    model_dir: Path,
    device: DeviceType,
    llm,
    pixel_values_slices: List[torch.Tensor],
    tgt_sizes: torch.Tensor,
    bounds: torch.Tensor,
    preproc_cfg: dict,
    debug: bool,
) -> Tuple[List[int], torch.Tensor]:
    # Reuse the vision+resampler builder from scripts/minicpmv_chat.py.
    from minicpmv_chat import _build_minicpmv_vision_model  # noqa: WPS433
    from libinfinicore_infer import DataType

    llm_torch_dt = llm.meta.torch_dtype_logits
    llm_dt = llm.meta.dt_logits
    vision_force_f32 = bool(int(__import__("os").environ.get("MINICPMV_VISION_FORCE_F32", "0")))
    vision_torch_dt = torch.float32 if vision_force_f32 else llm_torch_dt
    vision_dt = DataType.INFINI_DTYPE_F32 if vision_force_f32 else llm_dt

    vision_model, vision_handle, vision_meta, _keepalive = _build_minicpmv_vision_model(
        model_dir, vision_torch_dt, vision_dt, device
    )

    patch = int(preproc_cfg.get("patch_size", 14))
    slice_embeds: List[torch.Tensor] = []
    for i, x in enumerate(pixel_values_slices):
        th, tw = int(tgt_sizes[i][0].item()), int(tgt_sizes[i][1].item())
        seq_len = th * tw
        x = x.to(dtype=vision_torch_dt).contiguous()
        packed = x.unsqueeze(0).contiguous()
        if packed.shape != (1, 3, patch, seq_len * patch):
            raise RuntimeError(f"bad packed shape: {tuple(packed.shape)} for slice {i}")

        out = torch.empty(
            (vision_meta.resampler_meta.num_queries, vision_meta.resampler_meta.embed_dim),
            dtype=vision_torch_dt,
        )
        vision_model.infer_vision_resampler(vision_handle, packed.data_ptr(), seq_len, th, tw, out.data_ptr())
        if torch.isnan(out).any():
            raise RuntimeError(f"vision output contains NaNs (slice={i})")
        if out.dtype != llm_torch_dt:
            out = out.to(dtype=llm_torch_dt)
        slice_embeds.append(out.contiguous())

    # Flatten override positions and embeddings according to image_bound.
    override_pos_list: List[int] = []
    override_embed_list: List[torch.Tensor] = []
    for i in range(int(bounds.shape[0])):
        s = int(bounds[i][0].item())
        e = int(bounds[i][1].item())
        if e - s != int(vision_meta.resampler_meta.num_queries):
            raise RuntimeError(f"unexpected bound length: {e-s} (expected {vision_meta.resampler_meta.num_queries})")
        override_pos_list.extend(list(range(s, e)))
        override_embed_list.append(slice_embeds[i])
    override_embeds = torch.cat(override_embed_list, dim=0).contiguous()

    if debug:
        print(
            "DEBUG override_embeds stats:",
            float(override_embeds.float().abs().max().item()),
            float(override_embeds.float().abs().mean().item()),
            override_embeds.dtype,
        )
    return override_pos_list, override_embeds


def _pad_tokens_to_len(tokens: torch.Tensor, target_len: int, dvoc: int, seed: int) -> torch.Tensor:
    if int(tokens.numel()) > int(target_len):
        raise ValueError(f"prompt_len={target_len} is smaller than base prompt tokens {int(tokens.numel())}")
    if int(tokens.numel()) == int(target_len):
        return tokens
    rng = random.Random(int(seed))
    extra = [int(rng.randrange(int(dvoc))) for _ in range(int(target_len) - int(tokens.numel()))]
    out = torch.cat([tokens.to(dtype=torch.int32), torch.tensor(extra, dtype=torch.int32)], dim=0).contiguous()
    return out


def _bench_one_bs(
    *,
    llm,
    model_instance,
    device: DeviceType,
    dev_ids,
    tokens: torch.Tensor,
    override_pos_single: List[int],
    override_embeds_single: torch.Tensor,
    bs: int,
    warmup_steps: int,
    decode_steps: int,
    compress_bin: str,
    compression_factor: int,
    min_seq_len: int,
    image_kv_len: int,
):
    ntok = int(tokens.numel())
    tokens_list = [int(x) for x in tokens.tolist()]

    # Pack tokens for bs requests.
    tokens_c = (c_uint * (bs * ntok))(*(tokens_list * bs))
    req_lens = (c_uint * bs)(*([ntok] * bs))
    req_pos0 = (c_uint * bs)(*([0] * bs))

    # Override positions and embeds (replicated across requests).
    override_pos_all: List[int] = []
    for r in range(bs):
        base = r * ntok
        override_pos_all.extend([base + int(p) for p in override_pos_single])
    override_pos_c = (c_uint * len(override_pos_all))(*override_pos_all)
    override_embeds_all = override_embeds_single.repeat(bs, 1).contiguous()

    # Allocate kv caches.
    kv_list = [
        llm.jiuge_model.create_kv_cache(
            llm.meta.nlayer,
            llm.meta.dctx,
            llm.meta.nkvh,
            llm.meta.dh,
            llm.meta.dh,
            llm.meta.dt_logits,
            device,
            dev_ids,
            1,
        )
        for _ in range(bs)
    ]
    kv_caches = (POINTER(KVCacheCStruct) * bs)(*kv_list)

    def free_kvs():
        for kv in kv_list:
            llm.jiuge_model.drop_kv_cache(kv)

    temperature = (c_float * bs)(*([1.0] * bs))
    topk = (c_uint * bs)(*([1] * bs))
    topp = (c_float * bs)(*([1.0] * bs))

    # ---------------- baseline (no compression) ----------------
    out = (c_uint * bs)()
    try:
        t0 = time.perf_counter()
        llm.jiuge_model.infer_batch_with_overrides(
            model_instance,
            tokens_c,
            bs * ntok,
            req_lens,
            bs,
            req_pos0,
            kv_caches,
            len(override_pos_all),
            override_pos_c,
            override_embeds_all.data_ptr(),
            temperature,
            topk,
            topp,
            out,
        )
        t1 = time.perf_counter()
        prefill_s = float(t1 - t0)

        last_tokens = [int(out[i]) for i in range(bs)]
        rope_pos = ntok

        for w in range(int(warmup_steps)):
            tokens_step = (c_uint * bs)(*last_tokens)
            req_lens_step = (c_uint * bs)(*([1] * bs))
            req_pos_step = (c_uint * bs)(*([rope_pos + w] * bs))
            out_step = (c_uint * bs)()
            llm.jiuge_model.infer_batch(
                model_instance,
                tokens_step,
                bs,
                req_lens_step,
                bs,
                req_pos_step,
                kv_caches,
                temperature,
                topk,
                topp,
                out_step,
            )
            last_tokens = [int(out_step[i]) for i in range(bs)]

        t2 = time.perf_counter()
        for s in range(int(decode_steps)):
            tokens_step = (c_uint * bs)(*last_tokens)
            req_lens_step = (c_uint * bs)(*([1] * bs))
            req_pos_step = (c_uint * bs)(*([rope_pos + warmup_steps + s] * bs))
            out_step = (c_uint * bs)()
            llm.jiuge_model.infer_batch(
                model_instance,
                tokens_step,
                bs,
                req_lens_step,
                bs,
                req_pos_step,
                kv_caches,
                temperature,
                topk,
                topp,
                out_step,
            )
            last_tokens = [int(out_step[i]) for i in range(bs)]
        t3 = time.perf_counter()
        decode_s = float(t3 - t2)
        baseline_tps = (bs * decode_steps) / decode_s if decode_s > 0 else 0.0
    finally:
        free_kvs()

    # ---------------- compression ----------------
    kv_list = [
        llm.jiuge_model.create_kv_cache(
            llm.meta.nlayer,
            llm.meta.dctx,
            llm.meta.nkvh,
            llm.meta.dh,
            llm.meta.dh,
            llm.meta.dt_logits,
            device,
            dev_ids,
            1,
        )
        for _ in range(bs)
    ]
    kv_caches = (POINTER(KVCacheCStruct) * bs)(*kv_list)

    def free_kvs2():
        for kv in kv_list:
            llm.jiuge_model.drop_kv_cache(kv)

    out = (c_uint * bs)()
    try:
        t0 = time.perf_counter()
        llm.jiuge_model.infer_batch_with_overrides(
            model_instance,
            tokens_c,
            bs * ntok,
            req_lens,
            bs,
            req_pos0,
            kv_caches,
            len(override_pos_all),
            override_pos_c,
            override_embeds_all.data_ptr(),
            temperature,
            topk,
            topp,
            out,
        )
        t1 = time.perf_counter()
        prefill_s2 = float(t1 - t0)

        last_tokens = [int(out[i]) for i in range(bs)]

        cfg = KVCompressionConfigCStruct(
            enable=1,
            compression_factor=int(compression_factor),
            min_seq_len=int(min_seq_len),
            image_kv_len=int(image_kv_len),
            weight_path=str(compress_bin).encode("utf-8"),
        )

        t2 = time.perf_counter()
        kv_pos0_list: List[int] = []
        for kv in kv_list:
            kv_pos0_list.append(int(llm.jiuge_model.compress_kv_cache_inplace(kv, int(ntok), cfg)))
        t3 = time.perf_counter()
        compress_s = float(t3 - t2)

        rope_pos = ntok
        for w in range(int(warmup_steps)):
            tokens_step = (c_uint * bs)(*last_tokens)
            req_lens_step = (c_uint * bs)(*([1] * bs))
            req_pos_step = (c_uint * bs)(*([rope_pos + w] * bs))
            kv_pos_step = (c_uint * bs)(*([kv_pos0_list[i] + w for i in range(bs)]))
            out_step = (c_uint * bs)()
            llm.jiuge_model.infer_batch_ex(
                model_instance,
                tokens_step,
                bs,
                req_lens_step,
                bs,
                req_pos_step,
                kv_pos_step,
                kv_caches,
                temperature,
                topk,
                topp,
                out_step,
            )
            last_tokens = [int(out_step[i]) for i in range(bs)]

        t4 = time.perf_counter()
        for s in range(int(decode_steps)):
            tokens_step = (c_uint * bs)(*last_tokens)
            req_lens_step = (c_uint * bs)(*([1] * bs))
            req_pos_step = (c_uint * bs)(*([rope_pos + warmup_steps + s] * bs))
            kv_pos_step = (c_uint * bs)(*([kv_pos0_list[i] + warmup_steps + s for i in range(bs)]))
            out_step = (c_uint * bs)()
            llm.jiuge_model.infer_batch_ex(
                model_instance,
                tokens_step,
                bs,
                req_lens_step,
                bs,
                req_pos_step,
                kv_pos_step,
                kv_caches,
                temperature,
                topk,
                topp,
                out_step,
            )
            last_tokens = [int(out_step[i]) for i in range(bs)]
        t5 = time.perf_counter()
        decode_s2 = float(t5 - t4)
        comp_tps = (bs * decode_steps) / decode_s2 if decode_s2 > 0 else 0.0
    finally:
        free_kvs2()

    return {
        "bs": int(bs),
        "prompt_len": int(ntok),
        "baseline_prefill_s": prefill_s,
        "baseline_decode_s": decode_s,
        "baseline_decode_tps": baseline_tps,
        "comp_prefill_s": prefill_s2,
        "comp_compress_s": compress_s,
        "comp_decode_s": decode_s2,
        "comp_decode_tps": comp_tps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", choices=["cpu", "nvidia"], default="nvidia")
    ap.add_argument("--batch-sizes", default="1,8,16,32,64")
    ap.add_argument("--model-dir", default="/data/huggingface/MiniCPM-V-2_6")
    ap.add_argument("--image", required=True)
    ap.add_argument("--question", default="图片是什么？")

    ap.add_argument("--prompt-len", type=int, default=0, help="Pad prompt tokens to this length (0 = use real prompt length).")
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--decode-steps", type=int, default=64)
    ap.add_argument("--warmup-steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument("--compress-bin", required=True)
    ap.add_argument("--compression-factor", type=int, default=5)
    ap.add_argument("--min-seq-len", type=int, default=2)
    ap.add_argument("--image-kv-len", type=int, default=0, help="Prefix tokens treated as image KV (0 = auto from image_bound).")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    torch.set_default_device("cpu")

    device = DeviceType.DEVICE_TYPE_NVIDIA if args.dev == "nvidia" else DeviceType.DEVICE_TYPE_CPU
    dtype_override = torch.float16 if device == DeviceType.DEVICE_TYPE_NVIDIA else None

    from jiuge import JiugeForCauslLM

    model_dir = Path(args.model_dir)
    llm = JiugeForCauslLM(
        str(model_dir),
        device=device,
        ndev=1,
        max_tokens=int(args.max_tokens),
        dtype_override=dtype_override,
    )

    tokens, pixel_values_slices, tgt_sizes, bounds, preproc_cfg = _build_minicpmv_prompt_and_inputs(
        model_dir=model_dir,
        llm=llm,
        image_path=str(args.image),
        question=str(args.question),
        debug=bool(args.debug),
    )

    override_pos_single, override_embeds_single = _compute_override_embeds(
        model_dir=model_dir,
        device=device,
        llm=llm,
        pixel_values_slices=pixel_values_slices,
        tgt_sizes=tgt_sizes,
        bounds=bounds,
        preproc_cfg=preproc_cfg,
        debug=bool(args.debug),
    )

    image_kv_len = int(args.image_kv_len)
    if image_kv_len == 0:
        image_kv_len = int(bounds[:, 1].max().item())

    dvoc = int(llm.meta.dvoc)
    if int(args.prompt_len) > 0:
        tokens = _pad_tokens_to_len(tokens, int(args.prompt_len), dvoc, int(args.seed))

    batch_sizes = [int(x) for x in str(args.batch_sizes).split(",") if x.strip()]
    print("\n=== MiniCPM-V-2_6 (multimodal, real image) ===")
    print(
        f"image={args.image} prompt_len={int(tokens.numel())} decode_steps={int(args.decode_steps)} "
        f"compression_factor={int(args.compression_factor)} min_seq_len={int(args.min_seq_len)} image_kv_len={int(image_kv_len)}"
    )

    dev_ids = (c_int * 1)(0)
    results = []
    for bs in batch_sizes:
        r = _bench_one_bs(
            llm=llm,
            model_instance=llm.model_instance,
            device=device,
            dev_ids=dev_ids,
            tokens=tokens,
            override_pos_single=override_pos_single,
            override_embeds_single=override_embeds_single,
            bs=int(bs),
            warmup_steps=int(args.warmup_steps),
            decode_steps=int(args.decode_steps),
            compress_bin=str(args.compress_bin),
            compression_factor=int(args.compression_factor),
            min_seq_len=int(args.min_seq_len),
            image_kv_len=int(image_kv_len),
        )
        results.append(r)
        print(f"\n[bs={bs}]")
        print(
            f"baseline: prefill={r['baseline_prefill_s']:.3f}s decode={r['baseline_decode_s']:.3f}s decode_tps={r['baseline_decode_tps']:.2f}"
        )
        print(
            f"kv_compress: prefill={r['comp_prefill_s']:.3f}s compress={r['comp_compress_s']:.3f}s decode={r['comp_decode_s']:.3f}s decode_tps={r['comp_decode_tps']:.2f}"
        )


if __name__ == "__main__":
    main()

