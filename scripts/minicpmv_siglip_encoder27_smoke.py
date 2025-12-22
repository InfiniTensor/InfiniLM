import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
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
    keepalive["ln1_w"] = to_dt(
        _load_tensor(model_dir, weight_map, f"vpm.encoder.layers.{layer_idx}.layer_norm1.weight")
    )
    keepalive["ln1_b"] = to_dt(
        _load_tensor(model_dir, weight_map, f"vpm.encoder.layers.{layer_idx}.layer_norm1.bias")
    )
    keepalive["ln2_w"] = to_dt(
        _load_tensor(model_dir, weight_map, f"vpm.encoder.layers.{layer_idx}.layer_norm2.weight")
    )
    keepalive["ln2_b"] = to_dt(
        _load_tensor(model_dir, weight_map, f"vpm.encoder.layers.{layer_idx}.layer_norm2.bias")
    )
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
    nlayer = int(vision_cfg["num_hidden_layers"])

    vision_meta = MiniCPMVVisionMetaCStruct(
        patch_size=patch,
        vision_embed_dim=d,
        vision_num_layers=nlayer,
        vision_num_heads=nh,
        vision_intermediate_size=di,
        vision_layer_norm_eps=1e-6,
        vision_image_size=int(vision_cfg["image_size"]),
        vision_num_positions=4900,
    )
    resampler_meta = MiniCPMVResamplerMetaCStruct(
        num_queries=0,
        embed_dim=0,
        num_heads=0,
        kv_dim=0,
        layer_norm_eps=1e-6,
        max_patches_h=0,
        max_patches_w=0,
    )
    language_meta = MiniCPMVLanguageMetaCStruct(
        dt_logits=dt,
        nlayer=0,
        d=0,
        nh=0,
        nkvh=0,
        dh=0,
        di=0,
        dctx=0,
        dvoc=0,
        epsilon=1e-6,
        theta=1.0,
        end_token=0,
    )
    meta = MiniCPMVMetaCStruct(
        vision_meta=vision_meta, resampler_meta=resampler_meta, language_meta=language_meta
    )

    layers_keepalive = {}
    layers = []
    for i in range(nlayer):
        lw, ka = _make_layer_struct(model_dir, weight_map, i, torch_dt)
        layers.append(lw)
        for k, v in ka.items():
            layers_keepalive[f"l{i}_{k}"] = v

    layers_arr = (MiniCPMVSiglipLayerWeightsCStruct * nlayer)(*layers)

    layers_keepalive["post_ln_w"] = (
        _load_tensor(model_dir, weight_map, "vpm.post_layernorm.weight").detach().to(dtype=torch_dt).contiguous()
    )
    layers_keepalive["post_ln_b"] = (
        _load_tensor(model_dir, weight_map, "vpm.post_layernorm.bias").detach().to(dtype=torch_dt).contiguous()
    )

    weights = MiniCPMVWeightsCStruct()
    weights.vpm_patch_embedding_weight = 0
    weights.vpm_patch_embedding_bias = 0
    weights.vpm_position_embedding = 0
    weights.vpm_layers = layers_arr
    weights.vpm_post_layernorm_weight = layers_keepalive["post_ln_w"].data_ptr()
    weights.vpm_post_layernorm_bias = layers_keepalive["post_ln_b"].data_ptr()
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
    model.infer_siglip_encoder(model_handle, nlayer, x.data_ptr(), seq_len, out_cpp.data_ptr())

    # Torch reference: encoder layers -> post_layernorm
    from minicpmv_config.modeling_navit_siglip import SiglipVisionConfig, SiglipEncoderLayer

    cfg = SiglipVisionConfig(
        hidden_size=d,
        intermediate_size=di,
        num_hidden_layers=nlayer,
        num_attention_heads=nh,
        num_channels=3,
        image_size=vision_cfg["image_size"],
        patch_size=patch,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
    )

    torch_layers = []
    for i in range(nlayer):
        layer = SiglipEncoderLayer(cfg).to(dtype=torch_dt).eval()
        layer.load_state_dict(
            {
                "layer_norm1.weight": _load_tensor(model_dir, weight_map, f"vpm.encoder.layers.{i}.layer_norm1.weight"),
                "layer_norm1.bias": _load_tensor(model_dir, weight_map, f"vpm.encoder.layers.{i}.layer_norm1.bias"),
                "self_attn.q_proj.weight": _load_tensor(model_dir, weight_map, f"vpm.encoder.layers.{i}.self_attn.q_proj.weight"),
                "self_attn.q_proj.bias": _load_tensor(model_dir, weight_map, f"vpm.encoder.layers.{i}.self_attn.q_proj.bias"),
                "self_attn.k_proj.weight": _load_tensor(model_dir, weight_map, f"vpm.encoder.layers.{i}.self_attn.k_proj.weight"),
                "self_attn.k_proj.bias": _load_tensor(model_dir, weight_map, f"vpm.encoder.layers.{i}.self_attn.k_proj.bias"),
                "self_attn.v_proj.weight": _load_tensor(model_dir, weight_map, f"vpm.encoder.layers.{i}.self_attn.v_proj.weight"),
                "self_attn.v_proj.bias": _load_tensor(model_dir, weight_map, f"vpm.encoder.layers.{i}.self_attn.v_proj.bias"),
                "self_attn.out_proj.weight": _load_tensor(model_dir, weight_map, f"vpm.encoder.layers.{i}.self_attn.out_proj.weight"),
                "self_attn.out_proj.bias": _load_tensor(model_dir, weight_map, f"vpm.encoder.layers.{i}.self_attn.out_proj.bias"),
                "mlp.fc1.weight": _load_tensor(model_dir, weight_map, f"vpm.encoder.layers.{i}.mlp.fc1.weight"),
                "mlp.fc1.bias": _load_tensor(model_dir, weight_map, f"vpm.encoder.layers.{i}.mlp.fc1.bias"),
                "mlp.fc2.weight": _load_tensor(model_dir, weight_map, f"vpm.encoder.layers.{i}.mlp.fc2.weight"),
                "mlp.fc2.bias": _load_tensor(model_dir, weight_map, f"vpm.encoder.layers.{i}.mlp.fc2.bias"),
                "layer_norm2.weight": _load_tensor(model_dir, weight_map, f"vpm.encoder.layers.{i}.layer_norm2.weight"),
                "layer_norm2.bias": _load_tensor(model_dir, weight_map, f"vpm.encoder.layers.{i}.layer_norm2.bias"),
            },
            strict=True,
        )
        torch_layers.append(layer)

    with torch.no_grad():
        h = x.unsqueeze(0)
        for layer in torch_layers:
            h = layer(h, attention_mask=None)[0]
        w = layers_keepalive["post_ln_w"]
        b = layers_keepalive["post_ln_b"]
        out_ref = F.layer_norm(h, (d,), w, b, 1e-6).squeeze(0)

    diff = (out_cpp.float() - out_ref.float()).abs()
    print("siglip encoder27 smoke:")
    print("  dtype:", torch_dt)
    print("  out_cpp:", tuple(out_cpp.shape))
    print("  max_abs:", diff.max().item())
    print("  mean_abs:", diff.mean().item())

    model.destroy_model(model_handle)


if __name__ == "__main__":
    main()

