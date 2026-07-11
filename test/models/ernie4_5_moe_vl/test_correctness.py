"""Inference correctness test for ERNIE-4.5-VL-28B-A3B.

Covers the three required input modalities and compares InfiniLM output token
sequences against HuggingFace transformers (the reference) under greedy decoding.

Usage:
    python test/models/ernie4_5_moe_vl/test_correctness.py \
        --model /path/to/ERNIE-4.5-VL-28B-A3B-Thinking \
        --device nvidia \
        --image test/assets/demo.jpg \
        --video test/assets/demo.mp4

The HF reference path is gated behind --with-reference (needs the model loadable
by transformers). Without it, the script just runs InfiniLM and prints outputs.
Per the task rules, transformers is used ONLY as a test reference here; the
adapted model/processor code does not depend on it for inference.
"""

import argparse
import ctypes
import os


def _disable_maca_device_heap():
    """Set MACA device malloc heap size to 0 before any GPU allocation.

    On MetaX C500 (64 GB), the model weights consume nearly all VRAM, leaving
    too little for MACA to create its default 8 MB kernel-side heap.  Setting
    the limit to 0 disables the heap entirely; model inference does not use
    device-side malloc so this is safe.
    """
    for libname in ("libmcruntime.so", "libhcruntime.so"):
        try:
            lib = ctypes.CDLL(libname)
            # mcLimitMallocHeapSize / hcLimitMallocHeapSize = 2 (same as cudaLimitMallocHeapSize)
            ret = lib.mcDeviceSetLimit(2, ctypes.c_size_t(0))
            if ret == 0:
                print(f"[INFO] {libname}: mcDeviceSetLimit(MallocHeapSize, 0) OK")
                return
            # fallback: try hc variant
            ret2 = lib.hcDeviceSetLimit(2, ctypes.c_size_t(0))
            if ret2 == 0:
                print(f"[INFO] {libname}: hcDeviceSetLimit(MallocHeapSize, 0) OK")
                return
        except OSError:
            continue
    print("[WARN] could not set device malloc heap size to 0 (library not found)")


_disable_maca_device_heap()


def build_conversation(text, image=None, video=None):
    # InfiniLM framework format: resolve_multimodal_inputs expects type=="image"
    # with image_url holding the file path. The HF reference path may need its own
    # message format — adapt in run_reference if the installed processor differs.
    content = []
    if image is not None:
        content.append({"type": "image", "image_url": image})
    if video is not None:
        content.append({"type": "video", "video_url": video})
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


def run_infinilm(model_path, device, conversation, max_new_tokens, ignore_eos=False, tp=1,
                 max_cache_len=1024):
    from infinilm.llm.llm import LLM
    from infinilm.llm.sampling_params import SamplingParams

    model = LLM(
        model_path=os.path.expanduser(model_path),
        device=device,
        tensor_parallel_size=tp,
        cache_type="static",
        max_batch_size=1,
        max_tokens=max_new_tokens,
        # A video expands to thousands of vision tokens (min_frames=16 -> >=2400
        # for a 320x240 clip), so the prompt alone can exceed the text/image
        # default; raise --max-cache-len for video. default 4096 (224 MB) exceeds
        # free VRAM on C500.
        max_cache_len=max_cache_len,
        temperature=1.0,
        top_k=1,  # greedy
        top_p=1.0,
    )

    sp = SamplingParams(
        temperature=1.0,
        top_k=1,
        top_p=1.0,
        max_tokens=max_new_tokens,
        ignore_eos=ignore_eos,
    )
    outputs = model.chat(messages=[conversation], sampling_params=sp)
    return outputs


def run_reference(model_path, conversation, max_new_tokens):
    """Reference path: load ERNIE-4.5-VL with transformers and greedy-generate.

    Used ONLY as a correctness reference (task §4); the adapted InfiniLM
    inference path does not depend on transformers. Returns (token_ids, text).

    The checkpoint is loaded with trust_remote_code=True so its bundled
    processing_ernie4_5_vl / modeling code drives tokenization and the vision
    pipeline. The processor.apply_chat_template(..., return_dict=True) call is
    the unified multimodal entry: it renders text + image/video placeholders and
    attaches pixel values in one shot, so the same `conversation` used by
    InfiniLM feeds the reference unchanged.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    model_path = os.path.expanduser(model_path)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy, matches InfiniLM top_k=1 / temperature ignored
            num_beams=1,
        )
    output_ids = generated[0][prompt_len:].tolist()
    text = processor.decode(output_ids, skip_special_tokens=True)
    return output_ids, text


def _infinilm_text(infinilm_out):
    """Best-effort extraction of decoded text from LLM.chat output."""
    o = infinilm_out[0] if isinstance(infinilm_out, (list, tuple)) and infinilm_out else infinilm_out
    if isinstance(o, str):
        return o
    for attr in ("text", "generated_text", "outputs"):
        val = getattr(o, attr, None)
        if isinstance(val, str):
            return val
    return str(o)


def compare(infinilm_out, reference_ids, reference_text):
    """Compare InfiniLM output against the HF reference (task §4).

    Primary metric is exact token-sequence match; that requires InfiniLM to
    surface generated token ids here. Until LLM.chat exposes them, we compare on
    decoded text (exact match; semantic equivalence is left to manual review).
    """
    text = _infinilm_text(infinilm_out)
    if text.strip() == (reference_text or "").strip():
        print("[PASS] exact text match")
        return True
    print("[FAIL] output differs from reference")
    print(f"  InfiniLM : {text!r}")
    print(f"  Reference: {reference_text!r}  ({len(reference_ids)} ref tokens)")
    return False


CASES = [
    ("text",  dict(text="用一句话介绍你自己。")),
    ("image", dict(text="描述这张图片。", image="IMAGE_PATH")),
    ("video", dict(text="描述这段视频的内容。", video="VIDEO_PATH")),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--image", default=None)
    ap.add_argument("--video", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--max-cache-len", type=int, default=1024,
                    help="KV cache length; raise for video (>=3072) since a clip "
                         "expands to thousands of vision tokens.")
    ap.add_argument("--with-reference", action="store_true")
    ap.add_argument("--cases", default="text,image,video")
    ap.add_argument("--tp", type=int, default=1,
                    help="Tensor-parallel size. Use 2 on MetaX C500 ×2 (the 59GB "
                         "weights do not fit one 64GB card alongside activations/KV).")
    ap.add_argument("--ignore-eos", action="store_true",
                    help="Ignore EOS during generation to see what tokens follow (debug mode).")
    args = ap.parse_args()

    selected = set(args.cases.split(","))
    for name, kw in CASES:
        if name not in selected:
            continue
        if name == "image":
            if not args.image:
                print("[SKIP] image case: no --image provided")
                continue
            kw["image"] = args.image
        if name == "video":
            if not args.video:
                print("[SKIP] video case: no --video provided")
                continue
            kw["video"] = args.video

        print(f"\n===== case: {name} =====")
        conversation = build_conversation(**kw)
        infinilm_out = run_infinilm(
            args.model, args.device, conversation, args.max_new_tokens,
            ignore_eos=args.ignore_eos, tp=args.tp, max_cache_len=args.max_cache_len,
        )
        print(f"[InfiniLM] {infinilm_out}")

        if args.with_reference:
            ref_ids, ref_text = run_reference(args.model, conversation, args.max_new_tokens)
            print(f"[Reference] {ref_text}")
            compare(infinilm_out, ref_ids, ref_text)


if __name__ == "__main__":
    main()
