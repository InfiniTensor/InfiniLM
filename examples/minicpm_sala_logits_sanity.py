import argparse
import os

import ctypes
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import infinicore
from infinilm.distributed import DistConfig
from infinilm.infer_engine import InferEngine
from infinilm.modeling_utils import load_model_state_dict_by_file


def topk(t: torch.Tensor, k: int = 10):
    v, i = torch.topk(t, k)
    return list(zip(i.tolist(), v.tolist()))


def _infini_cpu_tensor_to_float32_torch(t: "infinicore.Tensor") -> torch.Tensor:
    # Minimal CPU readback utility (InfiniCore currently lacks a public to_torch()).
    dev = getattr(t, "device")
    if getattr(dev, "type", None) != "cpu":
        raise ValueError(f"expected CPU tensor, got device={dev}")

    numel = int(t.numel())
    ptr = int(t.data_ptr())

    if t.dtype == infinicore.bfloat16:
        buf = (ctypes.c_uint16 * numel).from_address(ptr)
        u16 = np.ctypeslib.as_array(buf)
        u32 = u16.astype(np.uint32) << 16
        f32 = u32.view(np.float32).reshape(t.shape)
        return torch.from_numpy(f32)
    if t.dtype == infinicore.float16:
        buf = (ctypes.c_uint16 * numel).from_address(ptr)
        u16 = np.ctypeslib.as_array(buf)
        f16 = u16.view(np.float16).reshape(t.shape)
        return torch.from_numpy(f16.astype(np.float32))
    if t.dtype == infinicore.float32:
        buf = (ctypes.c_float * numel).from_address(ptr)
        f32 = np.ctypeslib.as_array(buf).reshape(t.shape)
        return torch.from_numpy(f32.astype(np.float32, copy=False))

    raise ValueError(f"unsupported dtype for logits readback: {t.dtype}")


DEBUG_LOG_PATH = "/home/zenghua/repos/.cursor/debug-9146ea.log"


def _append_debug_log(entry: dict) -> None:
    import json
    # Always append (create file if missing). Clearing is handled by the harness.
    with open(DEBUG_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _register_hf_layer_hooks(hf_model, max_layers: int = 3):
    """
    Register forward hooks on the first `max_layers` HF transformer layers to log
    per-layer output stats (min/max/mean/l2) into the debug log.
    """
    import time

    layers_container = None
    # MiniCPM-SALA typically exposes layers under one of these paths; try them in order.
    candidate_paths = [
        ("transformer", "layers"),
        ("model", "layers"),
        ("layers",),
    ]
    for path in candidate_paths:
        obj = hf_model
        ok = True
        for attr in path:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok:
            layers_container = list(obj)
            break

    if layers_container is None:
        print("[HF DEBUG] Could not locate transformer layers for hooks; skipping HF per-layer stats.")
        return []

    hooks = []

    def make_hook(idx: int):
        def _hook(_module, _inputs, output):
            import torch

            # Normalize possible (tensor, ...) outputs.
            out = output[0] if isinstance(output, tuple) else output
            if not isinstance(out, torch.Tensor):
                return
            t = out.detach().float()
            if t.numel() == 0:
                return
            mn = float(t.min().item())
            mx = float(t.max().item())
            mean = float(t.mean().item())
            l2 = float(t.norm().item())
            _append_debug_log({
                "sessionId": "9146ea",
                "runId": "prefill",
                "hypothesisId": "HF_L",
                "location": "minicpm_sala_logits_sanity.py:hf_layer_output",
                "message": "HF layer output stats",
                "data": {
                    "layer": int(idx),
                    "min": mn,
                    "max": mx,
                    "mean": mean,
                    "l2": l2,
                },
                "timestamp": int(time.time() * 1000),
            })

        return _hook

    for idx, layer in enumerate(layers_container[:max_layers]):
        hooks.append(layer.register_forward_hook(make_hook(idx)))
    return hooks


@torch.no_grad()
def run_prefill_only(model_path: str, prompt: str, k: int) -> None:
    import os
    import time
    os.environ["INFINI_DEBUG_LOG"] = DEBUG_LOG_PATH
    os.environ["INFINI_DEBUG_ATTN_DUMP"] = "1"
    device = torch.device("cuda")

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    enc = tok(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc.get("attention_mask", None)
    if attn_mask is not None:
        attn_mask = attn_mask.to(device)

    # ---------- HF reference (single prefill) ----------
    hf = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=None,
        trust_remote_code=True,
    ).to(device)
    hf.eval()
    hf_hooks = _register_hf_layer_hooks(hf_model=hf, max_layers=3)
    hf_out = hf(input_ids=input_ids, attention_mask=attn_mask)
    hf_logits_last = hf_out.logits[0, -1].float().cpu()

    # ---------- InfiniLM (single prefill) ----------
    inf_dev = infinicore.device("cuda", 0)
    eng = InferEngine(
        model_path=model_path,
        device=inf_dev,
        distributed_config=DistConfig(1),
        enable_graph_compiling=False,
        attention_backend="default",
    )
    load_model_state_dict_by_file(eng, model_path, dtype=infinicore.bfloat16)

    bsz, seqlen = input_ids.shape
    assert bsz == 1
    pos = torch.arange(seqlen, dtype=torch.int64).view(1, seqlen)
    input_offsets = torch.tensor([0, seqlen], dtype=torch.int32)
    past = torch.tensor([0], dtype=torch.int32)
    total = torch.tensor([seqlen], dtype=torch.int32)

    inf_logits_last = eng.forward_logits(
        infinicore.from_torch(input_ids.cpu()),
        position_ids=infinicore.from_torch(pos),
        past_kv_lengths=infinicore.from_torch(past),
        total_kv_lengths=infinicore.from_torch(total),
        input_offsets=infinicore.from_torch(input_offsets),
        top_k=1,
        top_p=1.0,
        temperature=1.0,
    )
    inf_logits_last_t = _infini_cpu_tensor_to_float32_torch(inf_logits_last).reshape(-1).cpu()

    # ---------- Compare ----------
    diff = (inf_logits_last_t - hf_logits_last).abs()
    inf_norm = inf_logits_last_t.norm().item()
    hf_norm = hf_logits_last.norm().item()
    norm_ratio = inf_norm / (hf_norm + 1e-12)
    print("== logits sanity check (prefill only, last token) ==")
    print("prompt:", repr(prompt))
    print("seq_len:", seqlen)
    print("vocab:", hf_logits_last.numel())
    print("[DEBUG] logits norm: Inf=%.4f HF=%.4f ratio(Inf/HF)=%.4f" % (inf_norm, hf_norm, norm_ratio))
    print("abs_diff: max =", diff.max().item(), "mean =", diff.mean().item())
    print("HF   topk:", topk(hf_logits_last, k)[:10])
    print("Inf  topk:", topk(inf_logits_last_t, k)[:10])
    print("SANITY_ONELINE ratio=%.4f max_diff=%.4f mean_diff=%.4f" % (norm_ratio, diff.max().item(), diff.mean().item()))

    # #region agent log
    _append_debug_log({
        "sessionId": "9146ea",
        "runId": "prefill",
        "hypothesisId": "H0",
        "location": "minicpm_sala_logits_sanity.py:prefill_summary",
        "message": "Sanity summary",
        "data": {
            "prompt": prompt,
            "seq_len": int(seqlen),
            "vocab": int(hf_logits_last.numel()),
            "inf_norm": float(inf_norm),
            "hf_norm": float(hf_norm),
            "ratio_inf_hf": float(norm_ratio),
            "diff_max": float(diff.max().item()),
            "diff_mean": float(diff.mean().item()),
            "hf_top1": int(torch.argmax(hf_logits_last).item()),
            "inf_top1": int(torch.argmax(inf_logits_last_t).item()),
            "has_inf_attn_dump": bool(os.path.isfile("/tmp/inf_attn_out_layer0.bin")),
        },
        "timestamp": int(time.time() * 1000),
    })
    # #endregion

    # Ensure hooks are removed after this run to avoid side effects.
    for h in hf_hooks:
        h.remove()

    # #region agent log
    import json
    import time
    if os.path.isfile("/tmp/hf_attn_out_layer0.pt") and os.path.isfile("/tmp/inf_attn_out_layer0.bin"):
        hf_attn = torch.load("/tmp/hf_attn_out_layer0.pt")
        inf_raw = np.fromfile("/tmp/inf_attn_out_layer0.bin", dtype=np.float32)
        inf_attn = torch.from_numpy(inf_raw).reshape(hf_attn.shape)
        adiff = (inf_attn - hf_attn).abs()
        ratio = inf_attn.norm() / (hf_attn.norm() + 1e-12)
        data = {
            "shape": list(hf_attn.shape),
            "max_abs_diff": float(adiff.max()),
            "mean_abs_diff": float(adiff.mean()),
            "norm_ratio_inf_hf": float(ratio),
            "hf_min": float(hf_attn.min()),
            "hf_max": float(hf_attn.max()),
            "inf_min": float(inf_attn.min()),
            "inf_max": float(inf_attn.max()),
        }
        print("[DEBUG] Pre-gate attn layer0 compare:", data)
        _append_debug_log({
            "sessionId": "9146ea",
            "hypothesisId": "H1",
            "location": "minicpm_sala_logits_sanity.py:compare_attn",
            "message": "Pre-gate attn compare",
            "data": data,
            "timestamp": int(time.time() * 1000),
        })
    # #endregion


@torch.no_grad()
def run_prefill_decode1(model_path: str, prompt: str, k: int) -> None:
    device = torch.device("cuda")

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    enc = tok(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc.get("attention_mask", None)
    if attn_mask is not None:
        attn_mask = attn_mask.to(device)

    # ---------- HF reference ----------
    hf = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=None,
        trust_remote_code=True,
    ).to(device)
    hf.eval()

    # Prefill
    hf_out = hf(input_ids=input_ids, attention_mask=attn_mask)
    hf_logits_last_prefill_gpu = hf_out.logits[0, -1].float()
    hf_logits_last_prefill = hf_logits_last_prefill_gpu.cpu()

    # One-step decode (HF)
    next_token_hf = torch.argmax(hf_logits_last_prefill_gpu, dim=-1).view(1, 1).to(device)
    hf_out2 = hf(input_ids=torch.cat([input_ids, next_token_hf], dim=1), attention_mask=None)
    hf_logits_last_decode = hf_out2.logits[0, -1].float().cpu()

    # ---------- InfiniLM ----------
    inf_dev = infinicore.device("cuda", 0)
    eng = InferEngine(
        model_path=model_path,
        device=inf_dev,
        distributed_config=DistConfig(1),
        enable_graph_compiling=False,
        attention_backend="default",
    )
    load_model_state_dict_by_file(eng, model_path, dtype=infinicore.bfloat16)

    # Prefill in InfiniLM
    bsz, seqlen = input_ids.shape
    assert bsz == 1
    pos = torch.arange(seqlen, dtype=torch.int64).view(1, seqlen)
    input_offsets = torch.tensor([0, seqlen], dtype=torch.int32)
    past = torch.tensor([0], dtype=torch.int32)
    total = torch.tensor([seqlen], dtype=torch.int32)

    inf_logits_last_prefill = eng.forward_logits(
        infinicore.from_torch(input_ids.cpu()),
        position_ids=infinicore.from_torch(pos),
        past_kv_lengths=infinicore.from_torch(past),
        total_kv_lengths=infinicore.from_torch(total),
        input_offsets=infinicore.from_torch(input_offsets),
        top_k=1,
        top_p=1.0,
        temperature=1.0,
    )
    inf_logits_last_prefill_t = _infini_cpu_tensor_to_float32_torch(inf_logits_last_prefill).reshape(-1).cpu()

    # One-step decode in InfiniLM (append last token from InfiniLM prefill)
    next_token_inf = torch.argmax(inf_logits_last_prefill_t, dim=-1).view(1, 1).to(torch.int64)
    input_ids2 = torch.cat([input_ids.cpu(), next_token_inf], dim=1)
    bsz2, seqlen2 = input_ids2.shape
    pos2 = torch.arange(seqlen2, dtype=torch.int64).view(1, seqlen2)
    input_offsets2 = torch.tensor([0, seqlen2], dtype=torch.int32)
    past2 = torch.tensor([seqlen], dtype=torch.int32)
    total2 = torch.tensor([seqlen2], dtype=torch.int32)

    inf_logits_last_decode = eng.forward_logits(
        infinicore.from_torch(input_ids2),
        position_ids=infinicore.from_torch(pos2),
        past_kv_lengths=infinicore.from_torch(past2),
        total_kv_lengths=infinicore.from_torch(total2),
        input_offsets=infinicore.from_torch(input_offsets2),
        top_k=1,
        top_p=1.0,
        temperature=1.0,
    )
    inf_logits_last_decode_t = _infini_cpu_tensor_to_float32_torch(inf_logits_last_decode).reshape(-1).cpu()

    print("== logits sanity check (prefill + 1 decode) ==")
    print("prompt:", repr(prompt))
    print("seq_len prefill:", seqlen, "decode_len:", 1)

    diff_prefill = (inf_logits_last_prefill_t - hf_logits_last_prefill).abs()
    diff_decode = (inf_logits_last_decode_t - hf_logits_last_decode).abs()

    print("-- Prefill last token --")
    print("abs_diff: max =", diff_prefill.max().item(), "mean =", diff_prefill.mean().item())
    print("HF   topk:", topk(hf_logits_last_prefill, k)[:10])
    print("Inf  topk:", topk(inf_logits_last_prefill_t, k)[:10])

    print("-- Decode last token --")
    print("abs_diff: max =", diff_decode.max().item(), "mean =", diff_decode.mean().item())
    print("HF   topk:", topk(hf_logits_last_decode, k)[:10])
    print("Inf  topk:", topk(inf_logits_last_decode_t, k)[:10])


@torch.no_grad()
def run_decode_loop(model_path: str, prompt: str, k: int, steps: int) -> None:
    device = torch.device("cuda")

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    enc = tok(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    hf = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=None,
        trust_remote_code=True,
    ).to(device)
    hf.eval()

    # HF decode loop
    hf_ids = input_ids.clone()
    hf_topk_history = []
    for _ in range(steps):
        out = hf(input_ids=hf_ids)
        logits_last = out.logits[0, -1].float().cpu()
        hf_topk_history.append(topk(logits_last, k)[:10])
        next_token = torch.argmax(logits_last, dim=-1).view(1, 1).to(device)
        hf_ids = torch.cat([hf_ids, next_token], dim=1)

    # InfiniLM decode loop
    inf_dev = infinicore.device("cuda", 0)
    eng = InferEngine(
        model_path=model_path,
        device=inf_dev,
        distributed_config=DistConfig(1),
        enable_graph_compiling=False,
        attention_backend="default",
    )
    load_model_state_dict_by_file(eng, model_path, dtype=infinicore.bfloat16)

    # Prefill
    ids_inf = input_ids.cpu()
    bsz, seqlen = ids_inf.shape
    pos = torch.arange(seqlen, dtype=torch.int64).view(1, seqlen)
    input_offsets = torch.tensor([0, seqlen], dtype=torch.int32)
    past = torch.tensor([0], dtype=torch.int32)
    total = torch.tensor([seqlen], dtype=torch.int32)

    logits = eng.forward_logits(
        infinicore.from_torch(ids_inf),
        position_ids=infinicore.from_torch(pos),
        past_kv_lengths=infinicore.from_torch(past),
        total_kv_lengths=infinicore.from_torch(total),
        input_offsets=infinicore.from_torch(input_offsets),
        top_k=1,
        top_p=1.0,
        temperature=1.0,
    )
    logits_t = _infini_cpu_tensor_to_float32_torch(logits).reshape(-1).cpu()
    inf_topk_history = [topk(logits_t, k)[:10]]

    # Decode steps
    for step in range(steps - 1):
        next_token = torch.argmax(logits_t, dim=-1).view(1, 1).to(torch.int64)
        ids_inf = torch.cat([ids_inf, next_token], dim=1)
        bsz, seqlen = ids_inf.shape
        pos = torch.arange(seqlen, dtype=torch.int64).view(1, seqlen)
        input_offsets = torch.tensor([0, seqlen], dtype=torch.int32)
        past = torch.tensor([seqlen - 1], dtype=torch.int32)
        total = torch.tensor([seqlen], dtype=torch.int32)

        logits = eng.forward_logits(
            infinicore.from_torch(ids_inf),
            position_ids=infinicore.from_torch(pos),
            past_kv_lengths=infinicore.from_torch(past),
            total_kv_lengths=infinicore.from_torch(total),
            input_offsets=infinicore.from_torch(input_offsets),
            top_k=1,
            top_p=1.0,
            temperature=1.0,
        )
        logits_t = _infini_cpu_tensor_to_float32_torch(logits).reshape(-1).cpu()
        inf_topk_history.append(topk(logits_t, k)[:10])

    print("== logits sanity check (decode loop) ==")
    print("prompt:", repr(prompt))
    print("steps:", steps)
    for i in range(len(inf_topk_history)):
        print(f"[step {i}]")
        print("HF  topk:", hf_topk_history[i])
        print("Inf topk:", inf_topk_history[i])


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--prompt", type=str, default="How are you")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument(
        "--mode",
        type=str,
        default="prefill",
        choices=["prefill", "decode1", "decodeN"],
        help="Which sanity mode to run",
    )
    ap.add_argument(
        "--decode_steps",
        type=int,
        default=8,
        help="Number of decode steps for decodeN",
    )
    args = ap.parse_args()

    if args.mode == "prefill":
        run_prefill_only(args.model_path, args.prompt, args.k)
    elif args.mode == "decode1":
        run_prefill_decode1(args.model_path, args.prompt, args.k)
    else:
        run_decode_loop(args.model_path, args.prompt, args.k, args.decode_steps)


if __name__ == "__main__":
    # This script expects you to set up env vars (INFINI_ROOT/LD_LIBRARY_PATH)
    # similarly to examples/jiuge.py inside the container.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
