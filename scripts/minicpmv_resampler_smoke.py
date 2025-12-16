import json
import math
import os
from pathlib import Path

import torch
from safetensors.torch import safe_open

from libinfinicore_infer import (
    DataType,
    DeviceType,
    MiniCPMVLanguageMetaCStruct,
    MiniCPMVMetaCStruct,
    MiniCPMVModel,
    MiniCPMVResamplerMetaCStruct,
    MiniCPMVVisionMetaCStruct,
    MiniCPMVWeightsCStruct,
)


def _dtype_from_config(torch_dtype: str):
    if torch_dtype == "bfloat16":
        return torch.bfloat16, DataType.INFINI_DTYPE_BF16
    if torch_dtype == "float16":
        return torch.float16, DataType.INFINI_DTYPE_F16
    if torch_dtype == "float32":
        return torch.float32, DataType.INFINI_DTYPE_F32
    return torch.bfloat16, DataType.INFINI_DTYPE_BF16


def _load_tensor(model_dir: Path, weight_map: dict, key: str) -> torch.Tensor:
    filename = weight_map[key]
    full = model_dir / filename
    with safe_open(str(full), framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def main():
    model_dir = Path(os.environ.get("MINICPMV_MODEL_DIR", ""))
    if not model_dir:
        raise SystemExit("Set MINICPMV_MODEL_DIR to the HF model directory.")

    config = json.loads((model_dir / "config.json").read_text())
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]

    force_f32 = os.environ.get("MINICPMV_FORCE_F32", "0") == "1"
    torch_dt_logits, dt_logits = _dtype_from_config(config.get("torch_dtype", "bfloat16"))
    if force_f32:
        torch_dt_logits, dt_logits = torch.float32, DataType.INFINI_DTYPE_F32
    torch.set_default_dtype(torch.float32)

    # Build meta (only fields used by resampler path are required, but keep consistent).
    language_meta = MiniCPMVLanguageMetaCStruct(
        dt_logits=dt_logits,
        nlayer=config["num_hidden_layers"],
        d=config["hidden_size"],
        nh=config["num_attention_heads"],
        nkvh=config["num_key_value_heads"],
        dh=config["hidden_size"] // config["num_attention_heads"],
        di=config["intermediate_size"],
        dctx=config["max_position_embeddings"],
        dvoc=config["vocab_size"],
        epsilon=float(config["rms_norm_eps"]),
        theta=float(config["rope_theta"]),
        end_token=int(config.get("eos_token_id", 0)),
    )

    vision_cfg = config["vision_config"]
    vision_meta = MiniCPMVVisionMetaCStruct(
        patch_size=vision_cfg["patch_size"],
        vision_embed_dim=vision_cfg["hidden_size"],
        vision_num_layers=vision_cfg["num_hidden_layers"],
        vision_num_heads=vision_cfg["num_attention_heads"],
        vision_intermediate_size=vision_cfg["intermediate_size"],
        vision_layer_norm_eps=1e-6,
        vision_image_size=vision_cfg["image_size"],
        vision_num_positions=(vision_cfg["image_size"] // vision_cfg["patch_size"])
        * (vision_cfg["image_size"] // vision_cfg["patch_size"]),
    )

    resampler_meta = MiniCPMVResamplerMetaCStruct(
        num_queries=config["query_num"],
        embed_dim=config["hidden_size"],
        num_heads=config["hidden_size"] // 128,
        kv_dim=vision_cfg["hidden_size"],
        layer_norm_eps=1e-6,
        max_patches_h=70,
        max_patches_w=70,
    )

    meta = MiniCPMVMetaCStruct(
        vision_meta=vision_meta, resampler_meta=resampler_meta, language_meta=language_meta
    )

    # Load resampler weights only. Convert to dt_logits and pre-transpose for our C++ reference GEMM layout.
    keepalive = {}

    def to_dt(x: torch.Tensor) -> torch.Tensor:
        return x.detach().to(dtype=torch_dt_logits).contiguous()

    # resampler.query: [Q, D]
    keepalive["query"] = to_dt(_load_tensor(model_dir, weight_map, "resampler.query"))

    # kv_proj: torch Linear(kv_dim -> embed_dim) => weight [embed_dim, kv_dim]
    # C++ expects [kv_dim, embed_dim] for GEMM.
    kv_w = _load_tensor(model_dir, weight_map, "resampler.kv_proj.weight")
    keepalive["kv_proj_w_t"] = to_dt(kv_w.transpose(0, 1))

    # in_proj: [3D, D] => C++ expects [D, 3D]
    in_w = _load_tensor(model_dir, weight_map, "resampler.attn.in_proj_weight")
    keepalive["in_proj_w_t"] = to_dt(in_w.transpose(0, 1))
    keepalive["in_proj_b"] = to_dt(_load_tensor(model_dir, weight_map, "resampler.attn.in_proj_bias"))

    # out proj: torch Linear(D->D) weight [D, D] => C++ expects [D, D] (in x out), so transpose.
    out_w = _load_tensor(model_dir, weight_map, "resampler.attn.out_proj.weight")
    keepalive["out_proj_w_t"] = to_dt(out_w.transpose(0, 1))
    keepalive["out_proj_b"] = to_dt(_load_tensor(model_dir, weight_map, "resampler.attn.out_proj.bias"))

    for k in ["ln_q", "ln_kv", "ln_post"]:
        keepalive[f"{k}_w"] = to_dt(_load_tensor(model_dir, weight_map, f"resampler.{k}.weight"))
        keepalive[f"{k}_b"] = to_dt(_load_tensor(model_dir, weight_map, f"resampler.{k}.bias"))

    # proj: [D, D] used as x @ proj, no transpose.
    keepalive["proj"] = to_dt(_load_tensor(model_dir, weight_map, "resampler.proj"))

    weights = MiniCPMVWeightsCStruct()
    # Vision weights unused in this smoke (set null)
    weights.vpm_patch_embedding_weight = 0
    weights.vpm_patch_embedding_bias = 0
    weights.vpm_position_embedding = 0
    weights.vpm_layers = None
    weights.vpm_post_layernorm_weight = 0
    weights.vpm_post_layernorm_bias = 0

    weights.resampler_query = keepalive["query"].data_ptr()
    weights.resampler_kv_proj_weight = keepalive["kv_proj_w_t"].data_ptr()
    weights.resampler_attn_in_proj_weight = keepalive["in_proj_w_t"].data_ptr()
    weights.resampler_attn_in_proj_bias = keepalive["in_proj_b"].data_ptr()
    weights.resampler_attn_out_proj_weight = keepalive["out_proj_w_t"].data_ptr()
    weights.resampler_attn_out_proj_bias = keepalive["out_proj_b"].data_ptr()
    weights.resampler_ln_q_weight = keepalive["ln_q_w"].data_ptr()
    weights.resampler_ln_q_bias = keepalive["ln_q_b"].data_ptr()
    weights.resampler_ln_kv_weight = keepalive["ln_kv_w"].data_ptr()
    weights.resampler_ln_kv_bias = keepalive["ln_kv_b"].data_ptr()
    weights.resampler_ln_post_weight = keepalive["ln_post_w"].data_ptr()
    weights.resampler_ln_post_bias = keepalive["ln_post_b"].data_ptr()
    weights.resampler_proj = keepalive["proj"].data_ptr()

    # Language weights unused in this smoke (set minimal)
    weights.nlayer = 0
    weights.dt_norm = dt_logits
    weights.dt_mat = dt_logits
    weights.transpose_linear_weights = 0
    weights.input_embd = 0
    weights.output_norm = 0
    weights.output_embd = 0
    weights.attn_norm = None
    weights.attn_qkv = None
    weights.attn_qkv_b = None
    weights.attn_q_norm = None
    weights.attn_k_norm = None
    weights.attn_o = None
    weights.ffn_norm = None
    weights.ffn_gate_up = None
    weights.ffn_down = None

    model = MiniCPMVModel()
    dev_ids = (c_int * 1)(0)
    model_handle = model.create_model(meta, weights, DeviceType.DEVICE_TYPE_CPU, 1, dev_ids)

    # Build a no-padding test input: seq_len = tgt_h * tgt_w.
    tgt_h, tgt_w = 14, 14
    seq_len = tgt_h * tgt_w
    x = torch.randn((seq_len, vision_cfg["hidden_size"]), dtype=torch_dt_logits)
    out_cpp = torch.empty((config["query_num"], config["hidden_size"]), dtype=torch_dt_logits)

    model.infer_resampler(model_handle, x.data_ptr(), seq_len, tgt_h, tgt_w, out_cpp.data_ptr())

    # Torch reference
    from minicpmv_config.resampler import Resampler  # repo-local

    ref = Resampler(
        num_queries=config["query_num"],
        embed_dim=config["hidden_size"],
        num_heads=config["hidden_size"] // 128,
        kv_dim=vision_cfg["hidden_size"],
        adaptive=True,
    )
    ref.load_state_dict(
        {
            "query": _load_tensor(model_dir, weight_map, "resampler.query"),
            "kv_proj.weight": kv_w,
            "attn.in_proj_weight": in_w,
            "attn.in_proj_bias": _load_tensor(model_dir, weight_map, "resampler.attn.in_proj_bias"),
            "attn.out_proj.weight": out_w,
            "attn.out_proj.bias": _load_tensor(model_dir, weight_map, "resampler.attn.out_proj.bias"),
            "ln_q.weight": _load_tensor(model_dir, weight_map, "resampler.ln_q.weight"),
            "ln_q.bias": _load_tensor(model_dir, weight_map, "resampler.ln_q.bias"),
            "ln_kv.weight": _load_tensor(model_dir, weight_map, "resampler.ln_kv.weight"),
            "ln_kv.bias": _load_tensor(model_dir, weight_map, "resampler.ln_kv.bias"),
            "ln_post.weight": _load_tensor(model_dir, weight_map, "resampler.ln_post.weight"),
            "ln_post.bias": _load_tensor(model_dir, weight_map, "resampler.ln_post.bias"),
            "proj": _load_tensor(model_dir, weight_map, "resampler.proj"),
        },
        strict=True,
    )
    ref = ref.eval()

    with torch.no_grad():
        ref = ref.to(dtype=torch_dt_logits)
        out_ref = ref(
            x.unsqueeze(0),
            tgt_sizes=torch.tensor([[tgt_h, tgt_w]], dtype=torch.int32),
        )
        out_ref = out_ref.squeeze(0).to(torch_dt_logits)

    diff = (out_cpp.float() - out_ref.float()).abs()
    print("resampler smoke:")
    print("  dtype:", torch_dt_logits)
    print("  out_cpp:", tuple(out_cpp.shape))
    print("  max_abs:", diff.max().item())
    print("  mean_abs:", diff.mean().item())

    model.destroy_model(model_handle)


if __name__ == "__main__":
    from ctypes import c_int

    main()
