import argparse
import random
import time
from ctypes import POINTER, c_float, c_int, c_uint
from typing import List, Optional

import torch

from libinfinicore_infer import DeviceType, KVCacheCStruct, KVCompressionConfigCStruct

def _make_tokens(dvoc: int, prompt_len: int, image_prefix_len: int, image_token_id: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    image_prefix_len = max(0, min(int(image_prefix_len), int(prompt_len)))
    out = [int(image_token_id)] * image_prefix_len
    out += [int(rng.randrange(dvoc)) for _ in range(prompt_len - image_prefix_len)]
    return out


def _bench_one(
    *,
    model_name: str,
    jiuge_model,
    model_instance,
    meta,
    device: DeviceType,
    ndev: int,
    dev_ids,
    compress_bin: str,
    batch_sizes: List[int],
    prompt_len: int,
    decode_steps: int,
    compression_factor: int,
    min_seq_len: int,
    image_kv_len: int,
    image_token_id: int,
    seed: int,
    warmup_steps: int,
):
    print(f"\n=== {model_name} ===")
    print(
        f"prompt_len={prompt_len} decode_steps={decode_steps} "
        f"compression_factor={compression_factor} min_seq_len={min_seq_len} image_kv_len={image_kv_len}"
    )

    dvoc = int(meta.dvoc)
    nlayer = int(meta.nlayer)
    dctx = int(meta.dctx)
    nkvh = int(meta.nkvh)
    dh = int(meta.dh)
    dt_logits = meta.dt_logits

    total_steps = int(warmup_steps) + int(decode_steps)
    if prompt_len + total_steps > dctx:
        raise ValueError(
            f"prompt_len+warmup_steps+decode_steps={prompt_len+total_steps} exceeds dctx={dctx} "
            f"(pass larger --max-tokens to the model loader)"
        )

    cfg = KVCompressionConfigCStruct(
        enable=1,
        compression_factor=int(compression_factor),
        min_seq_len=int(min_seq_len),
        image_kv_len=int(image_kv_len),
        weight_path=str(compress_bin).encode("utf-8"),
    )

    prompt_tokens = _make_tokens(dvoc, prompt_len, image_kv_len, image_token_id, seed=seed)

    for bs in batch_sizes:
        print(f"\n[bs={bs}]")

        def alloc_kvs():
            kv_list = [
                jiuge_model.create_kv_cache(
                    nlayer,
                    dctx,
                    nkvh,
                    dh,
                    dh,
                    dt_logits,
                    device,
                    dev_ids,
                    ndev,
                )
                for _ in range(bs)
            ]
            kv_caches = (POINTER(KVCacheCStruct) * bs)(*kv_list)
            return kv_list, kv_caches

        def free_kvs(kv_list):
            for kv in kv_list:
                jiuge_model.drop_kv_cache(kv)

        temperature = (c_float * bs)(*([1.0] * bs))
        topk = (c_uint * bs)(*([1] * bs))
        topp = (c_float * bs)(*([1.0] * bs))

        # ---------------- baseline (no compression) ----------------
        kv_list, kv_caches = alloc_kvs()
        try:
            # Prefill
            tokens = (c_uint * (bs * prompt_len))(*(prompt_tokens * bs))
            req_lens = (c_uint * bs)(*([prompt_len] * bs))
            req_pos = (c_uint * bs)(*([0] * bs))
            out = (c_uint * bs)()
            t0 = time.perf_counter()
            jiuge_model.infer_batch(
                model_instance,
                tokens,
                bs * prompt_len,
                req_lens,
                bs,
                req_pos,
                kv_caches,
                temperature,
                topk,
                topp,
                out,
            )
            t1 = time.perf_counter()
            prefill_s = float(t1 - t0)

            last_tokens = [int(out[i]) for i in range(bs)]

            # Warmup decode
            rope_pos = prompt_len
            for w in range(int(warmup_steps)):
                tokens_step = (c_uint * bs)(*last_tokens)
                req_lens_step = (c_uint * bs)(*([1] * bs))
                req_pos_step = (c_uint * bs)(*([rope_pos + w] * bs))
                out_step = (c_uint * bs)()
                jiuge_model.infer_batch(
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

            # Timed decode
            t2 = time.perf_counter()
            for s in range(int(decode_steps)):
                tokens_step = (c_uint * bs)(*last_tokens)
                req_lens_step = (c_uint * bs)(*([1] * bs))
                req_pos_step = (c_uint * bs)(*([rope_pos + warmup_steps + s] * bs))
                out_step = (c_uint * bs)()
                jiuge_model.infer_batch(
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
            print(f"baseline: prefill={prefill_s:.3f}s decode={decode_s:.3f}s decode_tps={baseline_tps:.2f}")
        finally:
            free_kvs(kv_list)

        # ---------------- compression ----------------
        kv_list, kv_caches = alloc_kvs()
        try:
            # Prefill
            tokens = (c_uint * (bs * prompt_len))(*(prompt_tokens * bs))
            req_lens = (c_uint * bs)(*([prompt_len] * bs))
            req_pos = (c_uint * bs)(*([0] * bs))
            out = (c_uint * bs)()
            t0 = time.perf_counter()
            jiuge_model.infer_batch(
                model_instance,
                tokens,
                bs * prompt_len,
                req_lens,
                bs,
                req_pos,
                kv_caches,
                temperature,
                topk,
                topp,
                out,
            )
            t1 = time.perf_counter()
            prefill_s2 = float(t1 - t0)

            last_tokens = [int(out[i]) for i in range(bs)]

            # Compress each request KV in-place after prefill.
            t2 = time.perf_counter()
            kv_pos0: List[int] = []
            for kv in kv_list:
                new_len = int(jiuge_model.compress_kv_cache_inplace(kv, int(prompt_len), cfg))
                kv_pos0.append(new_len)
            t3 = time.perf_counter()
            compress_s = float(t3 - t2)

            # Warmup decode (KV-pos decoupled)
            rope_pos = prompt_len
            for w in range(int(warmup_steps)):
                tokens_step = (c_uint * bs)(*last_tokens)
                req_lens_step = (c_uint * bs)(*([1] * bs))
                req_pos_step = (c_uint * bs)(*([rope_pos + w] * bs))
                kv_pos_step = (c_uint * bs)(*([kv_pos0[i] + w for i in range(bs)]))
                out_step = (c_uint * bs)()
                jiuge_model.infer_batch_ex(
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

            # Timed decode
            t4 = time.perf_counter()
            for s in range(int(decode_steps)):
                tokens_step = (c_uint * bs)(*last_tokens)
                req_lens_step = (c_uint * bs)(*([1] * bs))
                req_pos_step = (c_uint * bs)(*([rope_pos + warmup_steps + s] * bs))
                kv_pos_step = (c_uint * bs)(*([kv_pos0[i] + warmup_steps + s for i in range(bs)]))
                out_step = (c_uint * bs)()
                jiuge_model.infer_batch_ex(
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
            print(
                f"kv_compress: prefill={prefill_s2:.3f}s compress={compress_s:.3f}s "
                f"decode={decode_s2:.3f}s decode_tps={comp_tps:.2f}"
            )
        finally:
            free_kvs(kv_list)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", choices=["cpu", "nvidia"], default="nvidia")
    ap.add_argument("--batch-sizes", default="1,32,64,128")
    ap.add_argument("--prompt-len", type=int, default=640)
    ap.add_argument("--decode-steps", type=int, default=64)
    ap.add_argument("--warmup-steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument("--compression-factor", type=int, default=5)
    ap.add_argument("--min-seq-len", type=int, default=2)

    ap.add_argument("--llava-model-dir", default="/data/huggingface/llava-1.5-7b-hf")
    ap.add_argument("--llava-compress-bin", default="./compress_ckpt/llava_mlp.bin")
    ap.add_argument("--llava-max-tokens", type=int, default=704)
    ap.add_argument("--llava-image-kv-len", type=int, default=576)

    ap.add_argument("--minicpmv-model-dir", default="/data/huggingface/MiniCPM-V-2_6")
    ap.add_argument("--minicpmv-compress-bin", default="./compress_ckpt/minicpmv_mlp.bin")
    ap.add_argument("--minicpmv-max-tokens", type=int, default=704)
    ap.add_argument("--minicpmv-image-kv-len", type=int, default=64)

    ap.add_argument("--model", choices=["llava", "minicpmv", "both"], default="both")
    args = ap.parse_args()

    torch.set_default_device("cpu")

    device = DeviceType.DEVICE_TYPE_NVIDIA if args.dev == "nvidia" else DeviceType.DEVICE_TYPE_CPU
    ndev = 1
    dev_ids = (c_int * 1)(0)

    batch_sizes = [int(x) for x in str(args.batch_sizes).split(",") if x.strip()]

    if args.model in {"llava", "both"}:
        from llava import LLaVAForCauslLM

        llava = LLaVAForCauslLM(
            args.llava_model_dir,
            device=device,
            ndev=ndev,
            max_tokens=int(args.llava_max_tokens),
        )
        _bench_one(
            model_name="LLaVA-1.5-7B (language only)",
            jiuge_model=llava.jiuge_model,
            model_instance=llava.language_model_instance,
            meta=llava.language_meta,
            device=device,
            ndev=ndev,
            dev_ids=dev_ids,
            compress_bin=args.llava_compress_bin,
            batch_sizes=batch_sizes,
            prompt_len=int(args.prompt_len),
            decode_steps=int(args.decode_steps),
            compression_factor=int(args.compression_factor),
            min_seq_len=int(args.min_seq_len),
            image_kv_len=int(args.llava_image_kv_len),
            image_token_id=int(llava.config.get("image_token_index", 32000)),
            seed=int(args.seed),
            warmup_steps=int(args.warmup_steps),
        )

    if args.model in {"minicpmv", "both"}:
        from jiuge import JiugeForCauslLM

        minicpmv = JiugeForCauslLM(
            args.minicpmv_model_dir,
            device=device,
            ndev=ndev,
            max_tokens=int(args.minicpmv_max_tokens),
            dtype_override=torch.float16 if device == DeviceType.DEVICE_TYPE_NVIDIA else None,
        )
        unk_id = int(getattr(minicpmv.tokenizer, "unk_token_id", 0))
        _bench_one(
            model_name="MiniCPM-V-2_6 (language only)",
            jiuge_model=minicpmv.jiuge_model,
            model_instance=minicpmv.model_instance,
            meta=minicpmv.meta,
            device=device,
            ndev=ndev,
            dev_ids=dev_ids,
            compress_bin=args.minicpmv_compress_bin,
            batch_sizes=batch_sizes,
            prompt_len=int(args.prompt_len),
            decode_steps=int(args.decode_steps),
            compression_factor=int(args.compression_factor),
            min_seq_len=int(args.min_seq_len),
            image_kv_len=int(args.minicpmv_image_kv_len),
            image_token_id=unk_id,
            seed=int(args.seed),
            warmup_steps=int(args.warmup_steps),
        )


if __name__ == "__main__":
    main()
