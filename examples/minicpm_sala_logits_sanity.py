import argparse
import os

import ctypes
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from infllmv2_loader import preload_infllmv2_if_available

preload_infllmv2_if_available()

import infinicore
from infinilm.distributed import DistConfig
from infinilm.infer_engine import InferEngine
from infinilm.cache import StaticKVCacheConfig
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


DEBUG_LOG_PATH = "/tmp/minicpm_sala_sanity_debug.log"


def _build_slope_tensor_sanity(nheads: int, dtype=torch.float32):
    """Same as HF MiniCPM-SALA _build_slope_tensor (used for Simple GLA decay)."""
    import math
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        closest = 2 ** math.floor(math.log2(n))
        return get_slopes_power_of_2(closest) + get_slopes(2 * closest)[0::2][: n - closest]
    return torch.tensor(get_slopes(nheads), dtype=dtype)


def _torch_simple_gla_recurrent_sanity(q, k, v, g_gamma, scale):
    """Reference Simple GLA recurrent: S = S*exp(g_gamma) + k^T v; o_t = (q_t*scale)@S. q,k,v (B,T,H,D)."""
    q = q.float().transpose(1, 2)   # (B, H, T, D)
    k = k.float().transpose(1, 2)
    v = v.float().transpose(1, 2)
    B, H, T, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5
    q = q * scale
    o = q.new_zeros(B, H, T, V)
    S = q.new_zeros(B, H, K, V)
    gate = g_gamma.float().exp()  # (H,)
    for i in range(T):
        key = k[:, :, i, :]
        value = v[:, :, i, :]
        kv = key.unsqueeze(-1) * value.unsqueeze(-2)
        S = S * gate.view(1, -1, 1, 1) + kv
        o[:, :, i, :] = (q[:, :, i, :].unsqueeze(-1) * S).sum(-2)
    return o.transpose(1, 2)  # (B, T, H, D)


def _append_debug_log(entry: dict) -> None:
    import json
    # Always append (create file if missing). Clearing is handled by the harness.
    with open(DEBUG_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _sync_infini_device() -> None:
    try:
        infinicore.sync_device()
    except Exception:
        # Some handoff points happen before an InfiniCore runtime exists.
        pass


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
            try:
                torch.save(t.cpu(), f"/tmp/hf_layer_out_{idx}.pt")
            except Exception:
                pass
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


def check_layer0_weight_match(model_path: str, hf_model, infini_engine=None) -> None:
    """
    Layer 0 is minicpm4 in the checkpoint; under FORCE_ALL_LIGHTNING both HF and InfiniLM
    use these same weights as 'placeholder' for the lightning (Simple GLA) path.
    Verify checkpoint vs HF match and that InfiniLM expects the same layer0 keys.
    """
    import glob
    from safetensors.torch import safe_open

    prefix = "model.layers.0.self_attn."
    ckpt_l0 = {}
    for p in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
        with safe_open(p, framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.startswith(prefix):
                    ckpt_l0[k] = f.get_tensor(k).float()

    if not ckpt_l0:
        print("[DEBUG] Layer0 weight check: no checkpoint keys for %s" % prefix)
        return

    # HF: under FORCE_ALL_LIGHTNING, layer 0 is MiniCPMAttention (minicpm4); same checkpoint keys.
    backbone = getattr(hf_model, "model", None) or getattr(hf_model, "transformer", None)
    if backbone is None:
        print("[DEBUG] Layer0 weight check: could not get HF backbone")
        return
    hf_layer0_attn = backbone.layers[0].self_attn

    print("[DEBUG] Layer0 (minicpm4 placeholder) weight match:")
    for key in sorted(ckpt_l0.keys()):
        suffix = key[len(prefix) :]
        parts = suffix.split(".")
        obj = hf_layer0_attn
        for p in parts[:-1]:
            obj = getattr(obj, p, None)
            if obj is None:
                break
        if obj is None:
            print("  HF missing: %s" % key)
            continue
        hf_param = getattr(obj, parts[-1], None)
        if hf_param is None or not hasattr(hf_param, "float"):
            print("  HF no tensor: %s" % key)
            continue
        ckpt_t = ckpt_l0[key]
        hf_t = hf_param.detach().float().cpu()
        if ckpt_t.shape != hf_t.shape:
            print("  shape mismatch %s: ckpt %s vs hf %s" % (key, list(ckpt_t.shape), list(hf_t.shape)))
            continue
        diff = (ckpt_t - hf_t).abs()
        print("  %s: shape=%s max_diff=%.6f mean_diff=%.6f" % (
            suffix, list(ckpt_t.shape), diff.max().item(), diff.mean().item()))
    print("  (checkpoint vs HF: should be ~0 when HF loads from same checkpoint)")

    # InfiniLM: expected keys must include all layer0 self_attn keys so they are loaded (no fake init).
    if infini_engine is not None:
        try:
            expected = set(infini_engine.state_dict_keyname())
        except Exception:
            expected = set()
        l0_expected = [k for k in expected if "layers.0.self_attn" in k]
        for ckpt_key in sorted(ckpt_l0.keys()):
            if ckpt_key not in expected:
                # Try without "model." prefix in case engine uses different top-level name
                alt = ckpt_key.replace("model.", "", 1) if ckpt_key.startswith("model.") else ckpt_key
                if alt not in expected:
                    print("  [WARN] InfiniLM may not load %s (not in expected keys)" % ckpt_key)
        print("  InfiniLM layer0 self_attn expected keys: %d (checkpoint layer0 self_attn: %d)" % (
            len(l0_expected), len(ckpt_l0)))


def _register_hf_layer0_attn_input_hook(hf_model):
    """Save layer0 input to self_attn (output of input_layernorm) to /tmp/hf_layer0_attn_input.pt."""
    for path in [("model", "layers"), ("transformer", "layers"), ("layers",)]:
        obj = hf_model
        ok = True
        for attr in path:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok and len(obj) > 0:
            layer0_input_norm = getattr(obj[0], "input_layernorm", None)
            if layer0_input_norm is None:
                return []
            def _hook(_module, _args, output):
                if isinstance(output, torch.Tensor) and output.numel() > 0:
                    try:
                        torch.save(output.detach().float().cpu(), "/tmp/hf_layer0_attn_input.pt")
                    except Exception:
                        pass
            return [layer0_input_norm.register_forward_hook(_hook)]
    return []


def _register_hf_embed_hook(hf_model):
    """Register a pre_hook on the first decoder layer to save layer0 input (embed output) to /tmp/hf_embed_out.pt."""
    for path in [("model", "layers"), ("transformer", "layers"), ("layers",)]:
        obj = hf_model
        ok = True
        for attr in path:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok and len(obj) > 0:
            first_layer = obj[0]

            def _pre_hook(_module, args):
                if args and isinstance(args[0], torch.Tensor) and args[0].numel() > 0:
                    try:
                        torch.save(args[0].detach().float().cpu(), "/tmp/hf_embed_out.pt")
                    except Exception:
                        pass
                return None

            return [first_layer.register_forward_pre_hook(_pre_hook)]
    return []


def _register_hf_final_hidden_hook(hf_model):
    """Register a hook on the backbone (transformer/model) to save final hidden state to /tmp/hf_final_hidden.pt."""
    backbone = None
    for name in ("model", "transformer"):
        if hasattr(hf_model, name):
            backbone = getattr(hf_model, name)
            break
    if backbone is None:
        return []

    def _hook(_module, _inputs, output):
        # Handle ModelOutput (has last_hidden_state) or tuple (output[0]) or raw tensor
        out = getattr(output, "last_hidden_state", None)
        if out is None:
            out = output[0] if isinstance(output, tuple) else output
        if isinstance(out, torch.Tensor) and out.numel() > 0:
            try:
                torch.save(out.detach().float().cpu(), "/tmp/hf_final_hidden.pt")
            except Exception:
                pass

    return [backbone.register_forward_hook(_hook)]


@torch.no_grad()
def run_prefill_only(model_path: str, prompt: str, k: int) -> None:
    import os
    import time
    os.environ["INFINI_DEBUG_LOG"] = DEBUG_LOG_PATH
    os.environ["INFINI_DEBUG_ATTN_DUMP"] = "1"
    hf_cuda_index = int(os.environ.get("HF_CUDA_INDEX", "0"))
    device = torch.device(f"cuda:{hf_cuda_index}")

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    enc = tok(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc.get("attention_mask", None)
    if attn_mask is not None:
        attn_mask = attn_mask.to(device)

    # ---------- HF reference (single prefill) ----------
    # Important: if InfLLM-v2 is enabled via LD_PRELOAD, it can break triton/ptxas subprocesses
    # during HF model load/forward (because LD_PRELOAD affects /bin/sh subprocesses).
    # Temporarily clear LD_PRELOAD for the entire HF reference segment.
    _saved_ld_preload = os.environ.pop("LD_PRELOAD", None)
    try:
        # Force eager attention when supported; fall back if transformers doesn't accept the kwarg.
        try:
            hf = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map=None,
                trust_remote_code=True,
                attn_implementation="eager",
            ).to(device)
        except TypeError:
            hf = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map=None,
                trust_remote_code=True,
            ).to(device)
        hf.eval()
        hf_hooks = _register_hf_layer_hooks(hf_model=hf, max_layers=3)
        hf_embed_hooks = _register_hf_embed_hook(hf)
        hf_layer0_attn_input_hooks = _register_hf_layer0_attn_input_hook(hf)
        hf_final_hooks = _register_hf_final_hidden_hook(hf)

        hf_out = hf(input_ids=input_ids, attention_mask=attn_mask)
        hf_logits_last = hf_out.logits[0, -1].float().cpu()
        # Save HF embed (layer0 input) for alignment check
        with torch.no_grad():
            hf_embed = hf.model.embed_tokens(input_ids) * getattr(hf.config, "scale_emb", 1.0)
            torch.save(hf_embed.detach().float().cpu(), "/tmp/hf_embed_out.pt")
        for h in hf_hooks:
            h.remove()
        for h in hf_embed_hooks:
            h.remove()
        for h in hf_layer0_attn_input_hooks:
            h.remove()
        for h in hf_final_hooks:
            h.remove()
    finally:
        if _saved_ld_preload is not None:
            os.environ["LD_PRELOAD"] = _saved_ld_preload

    # ---------- InfiniLM (single prefill) ----------
    inf_cuda_index = int(os.environ.get("INFINILM_CUDA_INDEX", "0"))
    inf_dev = infinicore.device("cuda", inf_cuda_index)
    eng = InferEngine(
        model_path=model_path,
        device=inf_dev,
        distributed_config=DistConfig(1),
        cache_config=StaticKVCacheConfig(max_batch_size=1, max_cache_len=2048),
        enable_graph_compiling=False,
        attention_backend="default",
    )
    load_model_state_dict_by_file(eng, model_path, dtype=infinicore.bfloat16)
    _sync_infini_device()

    bsz, seqlen = input_ids.shape
    assert bsz == 1
    pos = torch.arange(seqlen, dtype=torch.int64).view(1, seqlen)
    input_offsets = torch.tensor([0, seqlen], dtype=torch.int32)
    past = torch.tensor([0], dtype=torch.int32)
    total = torch.tensor([seqlen], dtype=torch.int32)
    cu_seqlens_prefill = torch.tensor([0, seqlen], dtype=torch.int32)

    # Workaround: use int32 input_ids (int64 H2D/copy can produce wrong indices and embed mismatch).
    input_ids_inf = input_ids.cpu().to(torch.int32)
    inf_logits_last = eng.forward_logits(
        infinicore.from_torch(input_ids_inf),
        position_ids=infinicore.from_torch(pos),
        past_kv_lengths=infinicore.from_torch(past),
        total_kv_lengths=infinicore.from_torch(total),
        input_offsets=infinicore.from_torch(input_offsets),
        cu_seqlens=infinicore.from_torch(cu_seqlens_prefill),
        top_k=1,
        top_p=1.0,
        temperature=1.0,
    )
    _sync_infini_device()
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

    # #region embed (layer0 input) alignment
    hf_embed = None
    if os.path.isfile("/tmp/hf_embed_out.pt"):
        hf_embed = torch.load("/tmp/hf_embed_out.pt")
    if hf_embed is not None and os.path.isfile("/tmp/inf_embed_out.bin"):
        inf_embed_raw = np.fromfile("/tmp/inf_embed_out.bin", dtype=np.float32)
        inf_embed = torch.from_numpy(inf_embed_raw).reshape(hf_embed.shape)
        embed_ratio = inf_embed.norm().item() / (hf_embed.norm().item() + 1e-12)
        embed_diff = (inf_embed - hf_embed).abs()
        print("[DEBUG] Embed (layer0 input) compare: norm_ratio=%.6f max_diff=%.6f mean_diff=%.6f shape=%s" % (
            embed_ratio, embed_diff.max().item(), embed_diff.mean().item(), list(hf_embed.shape)))
    elif hf_embed is not None:
        # Fallback: compute InfiniLM-style embed in Python (baked scale_emb) and compare to HF
        import json
        import glob
        from safetensors.torch import load_file as safe_load_file
        scale_emb = getattr(hf.config, "scale_emb", 1.0)
        embed_weight = None
        for p in glob.glob(os.path.join(model_path, "*.safetensors")):
            state = safe_load_file(p, device="cpu")
            if "model.embed_tokens.weight" in state:
                embed_weight = state["model.embed_tokens.weight"].float()
                break
        if embed_weight is not None:
            # InfiniLM loader bakes scale_emb into weight; HF does embed * scale_emb at forward.
            # So InfiniLM effective: lookup(W * scale_emb); HF: lookup(W) * scale_emb. Same result.
            w_baked = embed_weight * scale_emb
            ids = input_ids.cpu().long()
            inf_embed_py = torch.nn.functional.embedding(ids, w_baked).float()
            embed_ratio = inf_embed_py.norm().item() / (hf_embed.norm().item() + 1e-12)
            embed_diff = (inf_embed_py - hf_embed).abs()
            print("[DEBUG] Embed (layer0 input) compare [Py fallback]: norm_ratio=%.6f max_diff=%.6f mean_diff=%.6f shape=%s" % (
                embed_ratio, embed_diff.max().item(), embed_diff.mean().item(), list(hf_embed.shape)))
        else:
            print("[DEBUG] Embed compare: HF embed saved; inf_embed_out.bin missing and could not load embed weight")
    else:
        print("[DEBUG] Embed compare: missing hf_embed_out.pt")
    # #endregion

    # #region layer0 divergence: input to attention (after input_layernorm)
    hf_l0_in = None
    if os.path.isfile("/tmp/hf_layer0_attn_input.pt"):
        hf_l0_in = torch.load("/tmp/hf_layer0_attn_input.pt")
    if hf_l0_in is not None and os.path.isfile("/tmp/inf_layer0_attn_input.bin"):
        inf_l0_in_raw = np.fromfile("/tmp/inf_layer0_attn_input.bin", dtype=np.float32)
        inf_l0_in = torch.from_numpy(inf_l0_in_raw).reshape(hf_l0_in.shape)
        r = inf_l0_in.norm().item() / (hf_l0_in.norm().item() + 1e-12)
        d = (inf_l0_in - hf_l0_in).abs()
        print("[DEBUG] Layer0 attn input (after input_layernorm): norm_ratio=%.6f max_diff=%.6f mean_diff=%.6f" % (
            r, d.max().item(), d.mean().item()))
    elif hf_l0_in is not None:
        # Fallback: compute InfiniLM layer0 attn input = RMSNorm(embed_out) with checkpoint weight
        import glob
        from safetensors.torch import safe_open
        scale_emb = getattr(hf.config, "scale_emb", 1.0)
        embed_weight = None
        ln_weight = None
        for p in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
            with safe_open(p, framework="pt", device="cpu") as f:
                if embed_weight is None and "model.embed_tokens.weight" in f.keys():
                    embed_weight = f.get_tensor("model.embed_tokens.weight").float()
                if ln_weight is None and "model.layers.0.input_layernorm.weight" in f.keys():
                    ln_weight = f.get_tensor("model.layers.0.input_layernorm.weight").float()
            if embed_weight is not None and ln_weight is not None:
                break
        if embed_weight is not None and ln_weight is not None:
            w_emb = embed_weight * scale_emb
            ids = input_ids.cpu().long()
            embed_py = torch.nn.functional.embedding(ids, w_emb).float()
            # RMSNorm: x * weight / sqrt(mean(x^2) + eps)
            eps = getattr(hf.config, "rms_norm_eps", 1e-6)
            var = embed_py.pow(2).mean(-1, keepdim=True).add(eps)
            inf_l0_in = embed_py * ln_weight / torch.sqrt(var)
            r = inf_l0_in.norm().item() / (hf_l0_in.norm().item() + 1e-12)
            d = (inf_l0_in - hf_l0_in).abs()
            print("[DEBUG] Layer0 attn input (after input_layernorm) [Py fallback]: norm_ratio=%.6f max_diff=%.6f mean_diff=%.6f" % (
                r, d.max().item(), d.mean().item()))
        else:
            print("[DEBUG] Layer0 attn input: HF saved; inf dump/fallback not available")
    else:
        print("[DEBUG] Layer0 attn input: missing hf_layer0_attn_input.pt")
    # #endregion

    # #region layer0 GLA inputs (q,k,v) comparison
    def _compare_tensors(name, t_hf, t_inf_or_none):
        if t_inf_or_none is None:
            return
        if t_hf.shape != t_inf_or_none.shape:
            print("[DEBUG] Layer0 %s: shape mismatch HF %s vs Inf %s" % (name, list(t_hf.shape), list(t_inf_or_none.shape)))
            return
        r = t_inf_or_none.norm().item() / (t_hf.norm().item() + 1e-12)
        d = (t_inf_or_none - t_hf).abs()
        print("[DEBUG] Layer0 %s: norm_ratio=%.6f max_diff=%.6f mean_diff=%.6f" % (name, r, d.max().item(), d.mean().item()))

    if os.path.isfile("/tmp/hf_layer0_q.pt") and os.path.isfile("/tmp/hf_layer0_k.pt") and os.path.isfile("/tmp/hf_layer0_v.pt"):
        hf_q = torch.load("/tmp/hf_layer0_q.pt")
        hf_k = torch.load("/tmp/hf_layer0_k.pt")
        hf_v = torch.load("/tmp/hf_layer0_v.pt")
        if os.path.isfile("/tmp/inf_layer0_q.bin") and os.path.isfile("/tmp/inf_layer0_k.bin") and os.path.isfile("/tmp/inf_layer0_v.bin"):
            for name, hf_t, path in [("q", hf_q, "/tmp/inf_layer0_q.bin"), ("k", hf_k, "/tmp/inf_layer0_k.bin"), ("v", hf_v, "/tmp/inf_layer0_v.bin")]:
                inf_t = torch.from_numpy(np.fromfile(path, dtype=np.float32)).reshape(hf_t.shape)
                _compare_tensors(name, hf_t, inf_t)
        else:
            # Fallback: compute expected q,k,v from layer0 attn input + checkpoint weights; compare to HF (validates HF formation)
            import glob
            from safetensors.torch import safe_open
            if os.path.isfile("/tmp/hf_layer0_attn_input.pt"):
                attn_in = torch.load("/tmp/hf_layer0_attn_input.pt")
                B, S, _ = attn_in.shape
                n_h = getattr(hf.config, "num_attention_heads", 32)
                n_kv = getattr(hf.config, "num_key_value_heads", 2)
                head_dim = getattr(hf.config, "head_dim", 128)
                Wq = Wk = Wv = None
                for p in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
                    with safe_open(p, framework="pt", device="cpu") as f:
                        if "model.layers.0.self_attn.q_proj.weight" in f.keys():
                            Wq = f.get_tensor("model.layers.0.self_attn.q_proj.weight").float()
                            Wk = f.get_tensor("model.layers.0.self_attn.k_proj.weight").float()
                            Wv = f.get_tensor("model.layers.0.self_attn.v_proj.weight").float()
                            break
                if Wq is not None:
                    # (B,S,H*d) @ (H*d, H*d) -> (B,S,H*d); view (B,S,H,D)
                    q_py = (attn_in @ Wq.T).view(B, S, n_h, head_dim)
                    k_py = (attn_in @ Wk.T).view(B, S, n_kv, head_dim)
                    v_py = (attn_in @ Wv.T).view(B, S, n_kv, head_dim)
                    n_rep = n_h // n_kv
                    k_py = k_py[:, :, :, None, :].expand(B, S, n_kv, n_rep, head_dim).reshape(B, S, n_h, head_dim)
                    v_py = v_py[:, :, :, None, :].expand(B, S, n_kv, n_rep, head_dim).reshape(B, S, n_h, head_dim)
                    _compare_tensors("q (Py vs HF)", hf_q, q_py)
                    _compare_tensors("k (Py vs HF)", hf_k, k_py)
                    _compare_tensors("v (Py vs HF)", hf_v, v_py)
        print("[DEBUG] Layer0 GLA inputs: if q,k,v match then divergence is in GLA op or g_gamma.")
        # g_gamma: compare InfiniLM dump to expected decay (same as HF: _build_slope * -1)
        if os.path.isfile("/tmp/inf_layer0_g_gamma.bin"):
            n_heads = getattr(hf.config, "num_attention_heads", 32)
            expected_decay = _build_slope_tensor_sanity(n_heads) * (-1.0)
            inf_decay = torch.from_numpy(np.fromfile("/tmp/inf_layer0_g_gamma.bin", dtype=np.float32))
            if inf_decay.numel() == expected_decay.numel():
                r = inf_decay.norm().item() / (expected_decay.norm().item() + 1e-12)
                d = (inf_decay - expected_decay).abs()
                print("[DEBUG] Layer0 g_gamma (Inf vs HF formula): norm_ratio=%.6f max_diff=%.6f" % (r, d.max().item()))
            else:
                print("[DEBUG] Layer0 g_gamma: shape mismatch Inf %s vs expected %s" % (list(inf_decay.shape), list(expected_decay.shape)))
    # #endregion

    # Layer0 divergence summary
    print("[DEBUG] Layer0 divergence: embed and attn_input align ~1.0; "
          "pre-gate attn ~0.65 → see q/k/v comparison above to locate cause (proj vs GLA).")

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

        # Python GLA reference: run same recurrence on HF q,k,v; compare to HF and InfiniLM pre-gate.
        if os.path.isfile("/tmp/hf_layer0_q.pt") and os.path.isfile("/tmp/hf_layer0_k.pt") and os.path.isfile("/tmp/hf_layer0_v.pt"):
            _q = torch.load("/tmp/hf_layer0_q.pt")
            _k = torch.load("/tmp/hf_layer0_k.pt")
            _v = torch.load("/tmp/hf_layer0_v.pt")
            n_heads = getattr(hf.config, "num_attention_heads", 32)
            head_dim = getattr(hf.config, "head_dim", 128)
            decay = _build_slope_tensor_sanity(n_heads) * (-1.0)
            scale = head_dim ** (-0.5)
            py_gla = _torch_simple_gla_recurrent_sanity(_q, _k, _v, decay, scale)
            py_gla_flat = py_gla.reshape(hf_attn.shape)
            r_py_hf = py_gla_flat.norm().item() / (hf_attn.norm().item() + 1e-12)
            d_py_hf = (py_gla_flat - hf_attn).abs()
            print("[DEBUG] Python GLA (HF q,k,v) vs HF pre-gate: norm_ratio=%.6f max_diff=%.6f (expect ~1.0 if HF GLA consistent)" % (r_py_hf, d_py_hf.max().item()))
            if os.path.isfile("/tmp/inf_attn_out_layer0.bin"):
                inf_attn_flat = torch.from_numpy(np.fromfile("/tmp/inf_attn_out_layer0.bin", dtype=np.float32)).reshape(hf_attn.shape)
                r_py_inf = py_gla_flat.norm().item() / (inf_attn_flat.norm().item() + 1e-12)
                d_py_inf = (py_gla_flat - inf_attn_flat).abs()
                print("[DEBUG] Python GLA (HF q,k,v) vs InfiniLM pre-gate: norm_ratio=%.6f max_diff=%.6f" % (r_py_inf, d_py_inf.max().item()))
                print("[DEBUG] Root cause: if Py≈HF and Py≠Inf → divergence is InfiniLM q,k,v or InfiniLM GLA/g_gamma.")
    # #endregion

    # #region layer1 GLA inputs and pre-gate (align layer 1)
    if os.path.isfile("/tmp/hf_layer1_q.pt") and os.path.isfile("/tmp/hf_layer1_k.pt") and os.path.isfile("/tmp/hf_layer1_v.pt"):
        hf1_q = torch.load("/tmp/hf_layer1_q.pt")
        hf1_k = torch.load("/tmp/hf_layer1_k.pt")
        hf1_v = torch.load("/tmp/hf_layer1_v.pt")
        if os.path.isfile("/tmp/inf_layer1_q.bin") and os.path.isfile("/tmp/inf_layer1_k.bin") and os.path.isfile("/tmp/inf_layer1_v.bin"):
            for name, hf_t, path in [("q", hf1_q, "/tmp/inf_layer1_q.bin"), ("k", hf1_k, "/tmp/inf_layer1_k.bin"), ("v", hf1_v, "/tmp/inf_layer1_v.bin")]:
                inf_t = torch.from_numpy(np.fromfile(path, dtype=np.float32)).reshape(hf_t.shape)
                if inf_t.shape != hf_t.shape:
                    print("[DEBUG] Layer1 %s: shape mismatch HF %s vs Inf %s" % (name, list(hf_t.shape), list(inf_t.shape)))
                else:
                    r = inf_t.norm().item() / (hf_t.norm().item() + 1e-12)
                    d = (inf_t - hf_t).abs()
                    print("[DEBUG] Layer1 %s: norm_ratio=%.6f max_diff=%.6f mean_diff=%.6f" % (name, r, d.max().item(), d.mean().item()))
        if os.path.isfile("/tmp/hf_attn_out_layer1.pt") and os.path.isfile("/tmp/inf_attn_out_layer1.bin"):
            hf1_attn = torch.load("/tmp/hf_attn_out_layer1.pt")
            inf1_raw = np.fromfile("/tmp/inf_attn_out_layer1.bin", dtype=np.float32)
            inf1_attn = torch.from_numpy(inf1_raw).reshape(hf1_attn.shape)
            r1 = inf1_attn.norm().item() / (hf1_attn.norm().item() + 1e-12)
            d1 = (inf1_attn - hf1_attn).abs()
            print("[DEBUG] Layer1 pre-gate attn: norm_ratio=%.6f max_diff=%.6f mean_diff=%.6f" % (r1, d1.max().item(), d1.mean().item()))
    # #endregion

    # #region multi-layer and final hidden alignment
    print("[DEBUG] Decoder layer outputs (after emb) and final hidden alignment:")
    for layer_idx in range(3):
        hf_pt = f"/tmp/hf_layer_out_{layer_idx}.pt"
        inf_bin = f"/tmp/inf_layer_out_{layer_idx}.bin"
        if os.path.isfile(hf_pt) and os.path.isfile(inf_bin):
            hf_t = torch.load(hf_pt)
            inf_raw = np.fromfile(inf_bin, dtype=np.float32)
            inf_t = torch.from_numpy(inf_raw).reshape(hf_t.shape)
            ratio = inf_t.norm().item() / (hf_t.norm().item() + 1e-12)
            adiff = (inf_t - hf_t).abs()
            print("  layer%d: norm_ratio=%.6f max_diff=%.6f mean_diff=%.6f" % (
                layer_idx, ratio, adiff.max().item(), adiff.mean().item()))
        else:
            print("  layer%d: (missing %s or %s)" % (layer_idx, hf_pt, inf_bin))
    if os.path.isfile("/tmp/hf_final_hidden.pt") and os.path.isfile("/tmp/inf_final_hidden.bin"):
        hf_final = torch.load("/tmp/hf_final_hidden.pt")
        inf_final_raw = np.fromfile("/tmp/inf_final_hidden.bin", dtype=np.float32)
        inf_final = torch.from_numpy(inf_final_raw).reshape(hf_final.shape)
        ratio_final = inf_final.norm().item() / (hf_final.norm().item() + 1e-12)
        adiff_final = (inf_final - hf_final).abs()
        print("  final_hidden: norm_ratio=%.6f max_diff=%.6f mean_diff=%.6f" % (
            ratio_final, adiff_final.max().item(), adiff_final.mean().item()))
    else:
        print("  final_hidden: (missing hf or inf dump)")
    # #endregion

    # Layer 0 (minicpm4) weights: both sides use same checkpoint as placeholder when forced to lightning.
    check_layer0_weight_match(model_path, hf, eng)


@torch.no_grad()
def run_prefill_decode1(model_path: str, prompt: str, k: int) -> None:
    hf_cuda_index = int(os.environ.get("HF_CUDA_INDEX", "0"))
    device = torch.device(f"cuda:{hf_cuda_index}")

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    enc = tok(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc.get("attention_mask", None)
    if attn_mask is not None:
        attn_mask = attn_mask.to(device)

    # ---------- HF reference ----------
    _saved_ld_preload = os.environ.pop("LD_PRELOAD", None)
    try:
        try:
            hf = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map=None,
                trust_remote_code=True,
                attn_implementation="eager",
            ).to(device)
        except TypeError:
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
        # One-step decode (HF) is computed after Inf prefill so we can use
        # a shared decode input token (apples-to-apples).
    finally:
        if _saved_ld_preload is not None:
            os.environ["LD_PRELOAD"] = _saved_ld_preload

    # ---------- InfiniLM ----------
    inf_cuda_index = int(os.environ.get("INFINILM_CUDA_INDEX", "0"))
    inf_dev = infinicore.device("cuda", inf_cuda_index)
    eng = InferEngine(
        model_path=model_path,
        device=inf_dev,
        distributed_config=DistConfig(1),
        cache_config=StaticKVCacheConfig(max_batch_size=1, max_cache_len=2048),
        enable_graph_compiling=False,
        attention_backend="default",
    )
    load_model_state_dict_by_file(eng, model_path, dtype=infinicore.bfloat16)
    _sync_infini_device()

    # Prefill in InfiniLM
    bsz, seqlen = input_ids.shape
    assert bsz == 1
    pos = torch.arange(seqlen, dtype=torch.int64).view(1, seqlen)
    input_offsets = torch.tensor([0, seqlen], dtype=torch.int32)
    past = torch.tensor([0], dtype=torch.int32)
    total = torch.tensor([seqlen], dtype=torch.int32)
    # Match InferEngine.generate() metadata (batch=1): cu_seqlens = [0, past+cur_len]
    cu_seqlens_prefill = torch.tensor([0, seqlen], dtype=torch.int32)
    # Match `run_prefill_only()` workaround: use int32 input_ids to avoid
    # incorrect indices due to int64 H2D/copy behavior.
    input_ids_inf = input_ids.cpu().to(torch.int32)

    inf_logits_last_prefill = eng.forward_logits(
        infinicore.from_torch(input_ids_inf),
        position_ids=infinicore.from_torch(pos),
        past_kv_lengths=infinicore.from_torch(past),
        total_kv_lengths=infinicore.from_torch(total),
        input_offsets=infinicore.from_torch(input_offsets),
        cu_seqlens=infinicore.from_torch(cu_seqlens_prefill),
        top_k=1,
        top_p=1.0,
        temperature=1.0,
    )
    _sync_infini_device()
    inf_logits_last_prefill_t = _infini_cpu_tensor_to_float32_torch(inf_logits_last_prefill).reshape(-1).cpu()

    # One-step decode in InfiniLM.
    next_token_inf = torch.argmax(inf_logits_last_prefill_t, dim=-1).view(1, 1).to(torch.int64)
    if os.getenv("INFINI_DEBUG_PRINT_NEXT_TOKEN", "0") not in ("0", "", "false", "False"):
        try:
            vocab_size = getattr(tok, "vocab_size", None)
            if vocab_size is None and hasattr(hf, "config") and hasattr(hf.config, "vocab_size"):
                vocab_size = hf.config.vocab_size
            print(f"[DEBUG] next_token_inf={int(next_token_inf.item())} vocab_size={vocab_size}")
        except Exception:
            print(f"[DEBUG] next_token_inf={int(next_token_inf.item())}")
    input_ids2 = torch.cat([input_ids.cpu(), next_token_inf.cpu()], dim=1)
    past2 = torch.tensor([seqlen], dtype=torch.int32)
    total2 = torch.tensor([seqlen + 1], dtype=torch.int32)
    cu_seqlens_decode = torch.tensor([0, seqlen + 1], dtype=torch.int32)
    # Decode must pass only the new token to keep KV cache consistent.
    input_ids2_last = input_ids2[:, -1:].contiguous()
    pos2_last = torch.tensor([[seqlen]], dtype=torch.int64)
    input_offsets2 = torch.tensor([0, 1], dtype=torch.int32)

    # If we want HF decode hooks, also clear Inf decode dumps *before* the Inf decode call.
    use_hf_decode_hooks = os.environ.get("INFINI_DEBUG_ATTN_DUMP", "") not in ("", "0")
    if use_hf_decode_hooks:
        import glob
        for p in glob.glob("/tmp/inf_layer_out_*.bin"):
            try:
                os.remove(p)
            except Exception:
                pass

    # Important: keep decode indices tensor alive through the forward.
    # Creating the `infinicore.from_torch(...)` wrapper inline can lead to stale
    # indices for very small decode shapes ([1,1]) because decode kernels run
    # asynchronously; the embedding output (debug dump) may otherwise become
    # all zeros. Use the same pattern as prefill (`input_ids_inf`).
    input_ids2_last_int32 = input_ids2_last.to(torch.int32).contiguous()
    input_ids2_last_inf = infinicore.from_torch(input_ids2_last_int32)

    inf_logits_last_decode = eng.forward_logits(
        input_ids2_last_inf,
        position_ids=infinicore.from_torch(pos2_last),
        past_kv_lengths=infinicore.from_torch(past2),
        total_kv_lengths=infinicore.from_torch(total2),
        input_offsets=infinicore.from_torch(input_offsets2),
        cu_seqlens=infinicore.from_torch(cu_seqlens_decode),
        top_k=1,
        top_p=1.0,
        temperature=1.0,
    )
    _sync_infini_device()
    inf_logits_last_decode_t = _infini_cpu_tensor_to_float32_torch(inf_logits_last_decode).reshape(-1).cpu()

    # One-step decode (HF) on the same appended token we used for Inf.
    # Optional: dump per-layer HF outputs specifically for the decode step so we
    # can compare against InfiniLM decode dumps (which are last-token only).
    if use_hf_decode_hooks:
        import glob

        # Clear stale files from any earlier run.
        for p in glob.glob("/tmp/hf_layer_out_*.pt") + glob.glob("/tmp/hf_layer0_attn_input.pt"):
            try:
                os.remove(p)
            except Exception:
                pass

        hf_hooks = _register_hf_layer_hooks(hf_model=hf, max_layers=3)
        hf_layer0_attn_input_hooks = _register_hf_layer0_attn_input_hook(hf)
    else:
        hf_hooks = []
        hf_layer0_attn_input_hooks = []

    hf_out2 = hf(input_ids=torch.cat([input_ids, next_token_inf.to(device)], dim=1), attention_mask=None)
    if use_hf_decode_hooks:
        for h in hf_hooks:
            h.remove()
        for h in hf_layer0_attn_input_hooks:
            h.remove()
    hf_logits_last_decode = hf_out2.logits[0, -1].float().cpu()

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
    if torch.isnan(inf_logits_last_decode_t).any():
        print(
            "WARN: InfiniLM decode logits contain NaN (seen when prefill len is roughly >= 5 tokens on this build). "
            "Prefill gate above is still the primary correctness check; decode path needs further minicpm4/GLA work."
        )
    print("abs_diff: max =", diff_decode.max().item(), "mean =", diff_decode.mean().item())
    print("HF   topk:", topk(hf_logits_last_decode, k)[:10])
    print("Inf  topk:", topk(inf_logits_last_decode_t, k)[:10])


@torch.no_grad()
def run_decode_loop(model_path: str, prompt: str, k: int, steps: int) -> None:
    hf_cuda_index = int(os.environ.get("HF_CUDA_INDEX", "0"))
    device = torch.device(f"cuda:{hf_cuda_index}")

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    enc = tok(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    _saved_ld_preload = os.environ.pop("LD_PRELOAD", None)
    try:
        try:
            hf = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map=None,
                trust_remote_code=True,
                attn_implementation="eager",
            ).to(device)
        except TypeError:
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
        # Optional: dump HF hidden states for decode debugging.
        dump_hf = os.environ.get("DUMP_HF_DECODE_HIDDEN", "") not in ("", "0")
        dump_path = os.environ.get("DUMP_HF_DECODE_HIDDEN_PATH", "/tmp/hf_decode_hidden.pt")
        hf_hidden_dump = []
        for _ in range(steps):
            out = hf(input_ids=hf_ids, output_hidden_states=dump_hf, return_dict=True)
            logits_last = out.logits[0, -1].float().cpu()
            hf_topk_history.append(topk(logits_last, k)[:10])
            if dump_hf:
                # Save a small subset to keep files reasonable.
                # hidden_states is a tuple: (emb, layer1, ..., final)
                hs = out.hidden_states
                hf_hidden_dump.append(
                    {
                        "seq_len": hf_ids.shape[1],
                        "input_ids": hf_ids[0].detach().cpu(),
                        "emb": hs[0][0].detach().float().cpu(),
                        "layer0": hs[1][0].detach().float().cpu() if len(hs) > 1 else None,
                        "final": hs[-1][0].detach().float().cpu(),
                    }
                )
            next_token = torch.argmax(logits_last, dim=-1).view(1, 1).to(device)
            hf_ids = torch.cat([hf_ids, next_token], dim=1)
        if dump_hf:
            try:
                torch.save(hf_hidden_dump, dump_path)
                print(f"[DUMP] wrote HF hidden states to: {dump_path}")
            except Exception as e:
                print(f"[DUMP] failed to write HF hidden dump: {e}")
    finally:
        if _saved_ld_preload is not None:
            os.environ["LD_PRELOAD"] = _saved_ld_preload

    # InfiniLM decode loop
    # If C++ dumping is enabled, ensure we have a log path.
    if os.environ.get("MINICPM_SALA_DUMP_DECODE", "") not in ("", "0"):
        os.environ.setdefault("INFINI_DEBUG_LOG", "/tmp/minicpm_sala_decode_debug.log")
    inf_cuda_index = int(os.environ.get("INFINILM_CUDA_INDEX", "0"))
    inf_dev = infinicore.device("cuda", inf_cuda_index)
    eng = InferEngine(
        model_path=model_path,
        device=inf_dev,
        distributed_config=DistConfig(1),
        cache_config=StaticKVCacheConfig(max_batch_size=1, max_cache_len=2048),
        enable_graph_compiling=False,
        attention_backend="default",
    )
    load_model_state_dict_by_file(eng, model_path, dtype=infinicore.bfloat16)
    _sync_infini_device()

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
    _sync_infini_device()
    logits_t = _infini_cpu_tensor_to_float32_torch(logits).reshape(-1).cpu()
    inf_topk_history = [topk(logits_t, k)[:10]]

    # Decode steps: InfiniLM expects only the new token per step (seq_len=1) so KV cache
    # is updated correctly; passing full sequence would misalign the cache.
    for step in range(steps - 1):
        next_token = torch.argmax(logits_t, dim=-1).view(1, 1).to(torch.int64)
        ids_inf = torch.cat([ids_inf, next_token], dim=1)
        seqlen = ids_inf.shape[1]
        # Single-token input for decode (past tokens are in KV cache)
        decode_input = next_token.to(torch.int32)
        pos_decode = torch.tensor([[seqlen - 1]], dtype=torch.int64)
        input_offsets_decode = torch.tensor([0, 1], dtype=torch.int32)
        past = torch.tensor([seqlen - 1], dtype=torch.int32)
        total = torch.tensor([seqlen], dtype=torch.int32)

        # Keep decode indices tensor alive through the forward.
        # For tiny decode shapes ([1,1]), constructing `infinicore.from_torch(...)`
        # inline has previously caused stale indices -> embedding output becomes
        # zeros and decode logits diverge. Mirror the prefill int32 workaround.
        decode_input_int32 = decode_input.to(torch.int32).contiguous()
        decode_input_inf = infinicore.from_torch(decode_input_int32)

        logits = eng.forward_logits(
            decode_input_inf,
            position_ids=infinicore.from_torch(pos_decode),
            past_kv_lengths=infinicore.from_torch(past),
            total_kv_lengths=infinicore.from_torch(total),
            input_offsets=infinicore.from_torch(input_offsets_decode),
            top_k=1,
            top_p=1.0,
            temperature=1.0,
        )
        _sync_infini_device()
        logits_t = _infini_cpu_tensor_to_float32_torch(logits).reshape(-1).cpu()
        inf_topk_history.append(topk(logits_t, k)[:10])

    print("== logits sanity check (decode loop) ==")
    print("prompt:", repr(prompt))
    print("steps:", steps)
    for i in range(len(inf_topk_history)):
        print(f"[step {i}]")
        print("HF  topk:", hf_topk_history[i])
        print("Inf topk:", inf_topk_history[i])

    # One-prompt output comparison: decoded text and match
    prompt_len = input_ids.shape[1]
    hf_generated_ids = hf_ids[0, prompt_len:].cpu().tolist()
    inf_generated_ids = ids_inf[0, prompt_len:].cpu().tolist()
    hf_text = tok.decode(hf_generated_ids, skip_special_tokens=True)
    inf_text = tok.decode(inf_generated_ids, skip_special_tokens=True)
    print("\n== one-prompt output (both sides) ==")
    print("HF   generated:", repr(hf_text))
    print("Inf  generated:", repr(inf_text))
    if hf_generated_ids == inf_generated_ids:
        print("Match: exact (token ids identical)")
    else:
        first_diff = next((i for i in range(min(len(hf_generated_ids), len(inf_generated_ids))) if hf_generated_ids[i] != inf_generated_ids[i]), None)
        if first_diff is None:
            print("Match: length differs only (HF len=%d, Inf len=%d)" % (len(hf_generated_ids), len(inf_generated_ids)))
        else:
            print("Match: first divergence at generated token index %d (HF=%s Inf=%s)" % (first_diff, hf_generated_ids[first_diff], inf_generated_ids[first_diff]))
    print("Reasonable: HF and Inf outputs are above; judge fluency and relevance to prompt manually.")


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

    # Disabled so sanity uses compiled infllm-v2 in InfiniLM for layer0 (minicpm4).
    # To force all-lightning again: os.environ["MINICPM_SALA_FORCE_ALL_LIGHTNING"] = "1"
    # os.environ["MINICPM_SALA_FORCE_ALL_LIGHTNING"] = "1"

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
