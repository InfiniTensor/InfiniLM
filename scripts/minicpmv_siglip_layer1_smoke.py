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
    MiniCPMVSiglipLayerWeightsCStruct,
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


def _make_layer_struct(model_dir: Path, weight_map: dict, layer_idx: int, torch_dt) -> tuple:
    keepalive = {}

    def to_dt(x: torch.Tensor) -> torch.Tensor:
        return x.detach().to(dtype=torch_dt).contiguous()

    def t_weight(key: str) -> torch.Tensor:
        w = _load_tensor(model_dir, weight_map, key)
        return to_dt(w.transpose(0, 1))

    lw = MiniCPMVSiglipLayerWeightsCStruct()
    keepalive["ln1_w"] = to_dt(_load_tensor(model_dir, weight_map, f"vpm.encoder.layers.{layer_idx}.layer_norm1.weight"))
    keepalive["ln1_b"] = to_dt(_load_tensor(model_dir, weight_map, f"vpm.encoder.layers.{layer_idx}.layer_norm1.bias"))
    keepalive["ln2_w"] = to_dt(_load_tensor(model_dir, weight_map, f"vpm.encoder.layers.{layer_idx}.layer_norm2.weight"))
    keepalive["ln2_b"] = to_dt(_load_tensor(model_dir, weight_map, f"vpm.encoder.layers.{layer_idx}.layer_norm2.bias"))
    lw.layer_norm1_weight = keepalive["ln1_w"].data_ptr()
    lw.layer_norm1_bias = keepalive["ln1_b"].data_ptr()
    lw.layer_norm2_weight = keepalive["ln2_w"].data_ptr()
    lw.layer_norm2_bias = keepalive["ln2_b"].data_ptr()

    keepalive["q_w_t"] = t_weight(f"vpm.encoder.layers.{layer_idx}.self_attn.q_proj.weight")
    keepalive["k_w_t"] = t_weight(f"vpm.encoder.layers.{layer_idx}.self_attn.k_proj.weight")
    keepalive["v_w_t"] = t_weight(f"vpm.encoder.layers.{layer_idx}.self_attn.v_proj.weight")
    keepalive["o_w_t"] = t_weight(f"vpm.encoder.layers.{layer_idx}.self_attn.out_proj.weight")
    keepalive["q_b"] = to_dt(_load_tensor(model_dir, weight_map, f"vpm.encoder.layers.{layer_idx}.self_attn.q_proj.bias"))
    keepalive["k_b"] = to_dt(_load_tensor(model_dir, weight_map, f"vpm.encoder.layers.{layer_idx}.self_attn.k_proj.bias"))
    keepalive["v_b"] = to_dt(_load_tensor(model_dir, weight_map, f"vpm.encoder.layers.{layer_idx}.self_attn.v_proj.bias"))
    keepalive["o_b"] = to_dt(_load_tensor(model_dir, weight_map, f"vpm.encoder.layers.{layer_idx}.self_attn.out_proj.bias"))
    lw.q_weight = keepalive["q_w_t"].data_ptr()
    lw.q_bias = keepalive["q_b"].data_ptr()
    lw.k_weight = keepalive["k_w_t"].data_ptr()
    lw.k_bias = keepalive["k_b"].data_ptr()
    lw.v_weight = keepalive["v_w_t"].data_ptr()
    lw.v_bias = keepalive["v_b"].data_ptr()
    lw.out_weight = keepalive["o_w_t"].data_ptr()
    lw.out_bias = keepalive["o_b"].data_ptr()

    keepalive["fc1_w_t"] = t_weight(f"vpm.encoder.layers.{layer_idx}.mlp.fc1.weight")
    keepalive["fc2_w_t"] = t_weight(f"vpm.encoder.layers.{layer_idx}.mlp.fc2.weight")
    keepalive["fc1_b"] = to_dt(_load_tensor(model_dir, weight_map, f"vpm.encoder.layers.{layer_idx}.mlp.fc1.bias"))
    keepalive["fc2_b"] = to_dt(_load_tensor(model_dir, weight_map, f"vpm.encoder.layers.{layer_idx}.mlp.fc2.bias"))
    lw.fc1_weight = keepalive["fc1_w_t"].data_ptr()
    lw.fc1_bias = keepalive["fc1_b"].data_ptr()
    lw.fc2_weight = keepalive["fc2_w_t"].data_ptr()
    lw.fc2_bias = keepalive["fc2_b"].data_ptr()

    return lw, keepalive


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
    d = int(vision_cfg["hidden_size"])
    nh = int(vision_cfg["num_attention_heads"])
    di = int(vision_cfg["intermediate_size"])

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
        vision_embed_dim=d,
        vision_num_layers=vision_cfg["num_hidden_layers"],
        vision_num_heads=nh,
        vision_intermediate_size=di,
        vision_layer_norm_eps=1e-6,
        vision_image_size=vision_cfg["image_size"],
        vision_num_positions=(vision_cfg["image_size"] // patch) * (vision_cfg["image_size"] // patch),
    )
    resampler_meta = MiniCPMVResamplerMetaCStruct(
        num_queries=config["query_num"],
        embed_dim=config["hidden_size"],
        num_heads=config["hidden_size"] // 128,
        kv_dim=d,
        layer_norm_eps=1e-6,
        max_patches_h=70,
        max_patches_w=70,
    )
    meta = MiniCPMVMetaCStruct(
        vision_meta=vision_meta, resampler_meta=resampler_meta, language_meta=language_meta
    )

    lw0, ka0 = _make_layer_struct(model_dir, weight_map, 0, torch_dt)
    lw1, ka1 = _make_layer_struct(model_dir, weight_map, 1, torch_dt)
    keepalive = {**ka0, **{f"l1_{k}": v for k, v in ka1.items()}}
    layers_arr = (MiniCPMVSiglipLayerWeightsCStruct * 2)(lw0, lw1)

    weights = MiniCPMVWeightsCStruct()
    weights.vpm_patch_embedding_weight = 0
    weights.vpm_patch_embedding_bias = 0
    weights.vpm_position_embedding = 0
    weights.vpm_layers = layers_arr
    weights.vpm_post_layernorm_weight = 0
    weights.vpm_post_layernorm_bias = 0
    # Unused resampler weights
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
    # Unused language weights
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

    seq_len = 196
    x = torch.randn((seq_len, d), dtype=torch_dt)
    out_cpp = torch.empty((seq_len, d), dtype=torch_dt)
    model.infer_siglip_layer(model_handle, 1, x.data_ptr(), seq_len, out_cpp.data_ptr())

    # Torch reference layer1
    from minicpmv_config.modeling_navit_siglip import SiglipVisionConfig, SiglipEncoderLayer

    cfg = SiglipVisionConfig(
        hidden_size=d,
        intermediate_size=di,
        num_hidden_layers=vision_cfg["num_hidden_layers"],
        num_attention_heads=nh,
        num_channels=3,
        image_size=vision_cfg["image_size"],
        patch_size=patch,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
    )
    layer = SiglipEncoderLayer(cfg).to(dtype=torch_dt).eval()
    layer.load_state_dict(
        {
            "layer_norm1.weight": _load_tensor(model_dir, weight_map, "vpm.encoder.layers.1.layer_norm1.weight"),
            "layer_norm1.bias": _load_tensor(model_dir, weight_map, "vpm.encoder.layers.1.layer_norm1.bias"),
            "self_attn.q_proj.weight": _load_tensor(model_dir, weight_map, "vpm.encoder.layers.1.self_attn.q_proj.weight"),
            "self_attn.q_proj.bias": _load_tensor(model_dir, weight_map, "vpm.encoder.layers.1.self_attn.q_proj.bias"),
            "self_attn.k_proj.weight": _load_tensor(model_dir, weight_map, "vpm.encoder.layers.1.self_attn.k_proj.weight"),
            "self_attn.k_proj.bias": _load_tensor(model_dir, weight_map, "vpm.encoder.layers.1.self_attn.k_proj.bias"),
            "self_attn.v_proj.weight": _load_tensor(model_dir, weight_map, "vpm.encoder.layers.1.self_attn.v_proj.weight"),
            "self_attn.v_proj.bias": _load_tensor(model_dir, weight_map, "vpm.encoder.layers.1.self_attn.v_proj.bias"),
            "self_attn.out_proj.weight": _load_tensor(model_dir, weight_map, "vpm.encoder.layers.1.self_attn.out_proj.weight"),
            "self_attn.out_proj.bias": _load_tensor(model_dir, weight_map, "vpm.encoder.layers.1.self_attn.out_proj.bias"),
            "mlp.fc1.weight": _load_tensor(model_dir, weight_map, "vpm.encoder.layers.1.mlp.fc1.weight"),
            "mlp.fc1.bias": _load_tensor(model_dir, weight_map, "vpm.encoder.layers.1.mlp.fc1.bias"),
            "mlp.fc2.weight": _load_tensor(model_dir, weight_map, "vpm.encoder.layers.1.mlp.fc2.weight"),
            "mlp.fc2.bias": _load_tensor(model_dir, weight_map, "vpm.encoder.layers.1.mlp.fc2.bias"),
            "layer_norm2.weight": _load_tensor(model_dir, weight_map, "vpm.encoder.layers.1.layer_norm2.weight"),
            "layer_norm2.bias": _load_tensor(model_dir, weight_map, "vpm.encoder.layers.1.layer_norm2.bias"),
        },
        strict=True,
    )

    with torch.no_grad():
        out_ref = layer(x.unsqueeze(0), attention_mask=None)[0].squeeze(0)

    diff = (out_cpp.float() - out_ref.float()).abs()
    print("siglip layer1 smoke:")
    print("  dtype:", torch_dt)
    print("  out_cpp:", tuple(out_cpp.shape))
    print("  max_abs:", diff.max().item())
    print("  mean_abs:", diff.mean().item())

    model.destroy_model(model_handle)


if __name__ == "__main__":
    main()

