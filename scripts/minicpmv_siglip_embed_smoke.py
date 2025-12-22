import json
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
    torch_dt, dt = _dtype_from_config(config.get("torch_dtype", "bfloat16"))
    if force_f32:
        torch_dt, dt = torch.float32, DataType.INFINI_DTYPE_F32

    vision_cfg = config["vision_config"]
    patch = int(vision_cfg["patch_size"])
    vision_dim = int(vision_cfg["hidden_size"])

    # Meta: only vision fields are used by this smoke.
    language_meta = MiniCPMVLanguageMetaCStruct(
        dt_logits=dt,
        nlayer=0,
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
    vision_meta = MiniCPMVVisionMetaCStruct(
        patch_size=patch,
        vision_embed_dim=vision_dim,
        vision_num_layers=vision_cfg["num_hidden_layers"],
        vision_num_heads=vision_cfg["num_attention_heads"],
        vision_intermediate_size=vision_cfg["intermediate_size"],
        vision_layer_norm_eps=1e-6,
        vision_image_size=vision_cfg["image_size"],
        vision_num_positions=(vision_cfg["image_size"] // patch) * (vision_cfg["image_size"] // patch),
    )
    resampler_meta = MiniCPMVResamplerMetaCStruct(
        num_queries=config["query_num"],
        embed_dim=config["hidden_size"],
        num_heads=config["hidden_size"] // 128,
        kv_dim=vision_dim,
        layer_norm_eps=1e-6,
        max_patches_h=70,
        max_patches_w=70,
    )
    meta = MiniCPMVMetaCStruct(
        vision_meta=vision_meta, resampler_meta=resampler_meta, language_meta=language_meta
    )

    # Weights: only embedding weights required.
    keepalive = {}

    def to_dt(x: torch.Tensor) -> torch.Tensor:
        return x.detach().to(dtype=torch_dt).contiguous()

    keepalive["patch_w"] = to_dt(_load_tensor(model_dir, weight_map, "vpm.embeddings.patch_embedding.weight"))
    keepalive["patch_b"] = to_dt(_load_tensor(model_dir, weight_map, "vpm.embeddings.patch_embedding.bias"))
    keepalive["pos"] = to_dt(_load_tensor(model_dir, weight_map, "vpm.embeddings.position_embedding.weight"))

    weights = MiniCPMVWeightsCStruct()
    weights.vpm_patch_embedding_weight = keepalive["patch_w"].data_ptr()
    weights.vpm_patch_embedding_bias = keepalive["patch_b"].data_ptr()
    weights.vpm_position_embedding = keepalive["pos"].data_ptr()
    weights.vpm_layers = None
    weights.vpm_post_layernorm_weight = 0
    weights.vpm_post_layernorm_bias = 0

    # Resampler weights unused
    weights.resampler_query = 0
    weights.resampler_kv_proj_weight = 0
    weights.resampler_attn_in_proj_weight = 0
    weights.resampler_attn_in_proj_bias = 0
    weights.resampler_attn_out_proj_weight = 0
    weights.resampler_attn_out_proj_bias = 0
    weights.resampler_ln_q_weight = 0
    weights.resampler_ln_q_bias = 0
    weights.resampler_ln_kv_weight = 0
    weights.resampler_ln_kv_bias = 0
    weights.resampler_ln_post_weight = 0
    weights.resampler_ln_post_bias = 0
    weights.resampler_proj = 0

    # Language weights unused
    weights.nlayer = 0
    weights.dt_norm = dt
    weights.dt_mat = dt
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
    from ctypes import c_int

    dev_ids = (c_int * 1)(0)
    model_handle = model.create_model(meta, weights, DeviceType.DEVICE_TYPE_CPU, 1, dev_ids)

    tgt_h, tgt_w = 30, 34
    seq_len = tgt_h * tgt_w
    pixel_values = torch.randn((1, 3, patch, seq_len * patch), dtype=torch_dt)
    out_cpp = torch.empty((seq_len, vision_dim), dtype=torch_dt)
    model.infer_siglip_embeddings(model_handle, pixel_values.data_ptr(), seq_len, tgt_h, tgt_w, out_cpp.data_ptr())

    # Torch reference: SiglipVisionEmbeddings forward.
    from minicpmv_config.modeling_navit_siglip import SiglipVisionConfig, SiglipVisionEmbeddings

    cfg = SiglipVisionConfig(
        hidden_size=vision_dim,
        intermediate_size=vision_cfg["intermediate_size"],
        num_hidden_layers=vision_cfg["num_hidden_layers"],
        num_attention_heads=vision_cfg["num_attention_heads"],
        num_channels=3,
        image_size=vision_cfg["image_size"],
        patch_size=patch,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
    )
    emb = SiglipVisionEmbeddings(cfg).to(dtype=torch_dt)
    emb.load_state_dict(
        {
            "patch_embedding.weight": _load_tensor(model_dir, weight_map, "vpm.embeddings.patch_embedding.weight"),
            "patch_embedding.bias": _load_tensor(model_dir, weight_map, "vpm.embeddings.patch_embedding.bias"),
            "position_embedding.weight": _load_tensor(model_dir, weight_map, "vpm.embeddings.position_embedding.weight"),
        },
        strict=True,
    )
    emb = emb.eval()

    with torch.no_grad():
        patch_attn_mask = torch.ones((1, 1, seq_len), dtype=torch.bool)
        tgt_sizes = torch.tensor([[tgt_h, tgt_w]], dtype=torch.int32)
        out_ref = emb(pixel_values, patch_attn_mask, tgt_sizes=tgt_sizes).squeeze(0)

    diff = (out_cpp.float() - out_ref.float()).abs()
    print("siglip embeddings smoke:")
    print("  dtype:", torch_dt)
    print("  out_cpp:", tuple(out_cpp.shape))
    print("  max_abs:", diff.max().item())
    print("  mean_abs:", diff.mean().item())

    model.destroy_model(model_handle)


if __name__ == "__main__":
    main()

