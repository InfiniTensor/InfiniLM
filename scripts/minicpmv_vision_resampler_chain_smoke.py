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
    if key not in weight_map:
        if key.endswith(".weight") and key[: -len(".weight")] in weight_map:
            key = key[: -len(".weight")]
        elif (key + ".weight") in weight_map:
            key = key + ".weight"
    filename = weight_map[key]
    full = model_dir / filename
    with safe_open(str(full), framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def _make_siglip_layer_struct(model_dir: Path, weight_map: dict, layer_idx: int, torch_dt) -> tuple:
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

    # Resampler meta is tied to language meta in current C++ reference.
    language_meta = MiniCPMVLanguageMetaCStruct(
        dt_logits=dt,
        nlayer=0,
        d=int(config["hidden_size"]),
        nh=int(config["num_attention_heads"]),
        nkvh=int(config["num_key_value_heads"]),
        dh=int(config["hidden_size"] // config["num_attention_heads"]),
        di=int(config["intermediate_size"]),
        dctx=int(config["max_position_embeddings"]),
        dvoc=int(config["vocab_size"]),
        epsilon=float(config["rms_norm_eps"]),
        theta=float(config["rope_theta"]),
        end_token=int(config["eos_token_id"]),
    )

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
        num_queries=int(config["query_num"]),
        embed_dim=int(config["hidden_size"]),
        num_heads=int(config["num_attention_heads"]),
        kv_dim=d,
        layer_norm_eps=1e-6,
        max_patches_h=70,
        max_patches_w=70,
    )
    meta = MiniCPMVMetaCStruct(
        vision_meta=vision_meta, resampler_meta=resampler_meta, language_meta=language_meta
    )

    keepalive = {}

    # Vision weights
    keepalive["patch_w"] = _load_tensor(model_dir, weight_map, "vpm.embeddings.patch_embedding.weight").detach().to(dtype=torch_dt).contiguous()
    keepalive["patch_b"] = _load_tensor(model_dir, weight_map, "vpm.embeddings.patch_embedding.bias").detach().to(dtype=torch_dt).contiguous()
    keepalive["pos_emb"] = _load_tensor(model_dir, weight_map, "vpm.embeddings.position_embedding.weight").detach().to(dtype=torch_dt).contiguous()

    layers = []
    for i in range(nlayer):
        lw, ka = _make_siglip_layer_struct(model_dir, weight_map, i, torch_dt)
        layers.append(lw)
        for k, v in ka.items():
            keepalive[f"l{i}_{k}"] = v
    layers_arr = (MiniCPMVSiglipLayerWeightsCStruct * nlayer)(*layers)

    keepalive["post_ln_w"] = _load_tensor(model_dir, weight_map, "vpm.post_layernorm.weight").detach().to(dtype=torch_dt).contiguous()
    keepalive["post_ln_b"] = _load_tensor(model_dir, weight_map, "vpm.post_layernorm.bias").detach().to(dtype=torch_dt).contiguous()

    # Resampler weights (transpose to [in, out] layout)
    def t(key: str) -> torch.Tensor:
        return _load_tensor(model_dir, weight_map, key).detach().to(dtype=torch_dt).transpose(0, 1).contiguous()

    keepalive["res_kv_proj_w_t"] = t("resampler.kv_proj.weight")
    keepalive["res_in_w_t"] = t("resampler.attn.in_proj_weight")
    keepalive["res_out_w_t"] = t("resampler.attn.out_proj.weight")
    keepalive["res_in_b"] = _load_tensor(model_dir, weight_map, "resampler.attn.in_proj_bias").detach().to(dtype=torch_dt).contiguous()
    keepalive["res_out_b"] = _load_tensor(model_dir, weight_map, "resampler.attn.out_proj.bias").detach().to(dtype=torch_dt).contiguous()
    keepalive["res_query"] = _load_tensor(model_dir, weight_map, "resampler.query").detach().to(dtype=torch_dt).contiguous()
    # proj is used as `x @ proj` in HF (no transpose).
    keepalive["res_proj"] = _load_tensor(model_dir, weight_map, "resampler.proj").detach().to(dtype=torch_dt).contiguous()

    for name in ["ln_q", "ln_kv", "ln_post"]:
        keepalive[f"{name}_w"] = _load_tensor(model_dir, weight_map, f"resampler.{name}.weight").detach().to(dtype=torch_dt).contiguous()
        keepalive[f"{name}_b"] = _load_tensor(model_dir, weight_map, f"resampler.{name}.bias").detach().to(dtype=torch_dt).contiguous()

    weights = MiniCPMVWeightsCStruct()
    weights.vpm_patch_embedding_weight = keepalive["patch_w"].data_ptr()
    weights.vpm_patch_embedding_bias = keepalive["patch_b"].data_ptr()
    weights.vpm_position_embedding = keepalive["pos_emb"].data_ptr()
    weights.vpm_layers = layers_arr
    weights.vpm_post_layernorm_weight = keepalive["post_ln_w"].data_ptr()
    weights.vpm_post_layernorm_bias = keepalive["post_ln_b"].data_ptr()

    weights.resampler_query = keepalive["res_query"].data_ptr()
    weights.resampler_kv_proj_weight = keepalive["res_kv_proj_w_t"].data_ptr()
    weights.resampler_attn_in_proj_weight = keepalive["res_in_w_t"].data_ptr()
    weights.resampler_attn_in_proj_bias = keepalive["res_in_b"].data_ptr()
    weights.resampler_attn_out_proj_weight = keepalive["res_out_w_t"].data_ptr()
    weights.resampler_attn_out_proj_bias = keepalive["res_out_b"].data_ptr()
    weights.resampler_ln_q_weight = keepalive["ln_q_w"].data_ptr()
    weights.resampler_ln_q_bias = keepalive["ln_q_b"].data_ptr()
    weights.resampler_ln_kv_weight = keepalive["ln_kv_w"].data_ptr()
    weights.resampler_ln_kv_bias = keepalive["ln_kv_b"].data_ptr()
    weights.resampler_ln_post_weight = keepalive["ln_post_w"].data_ptr()
    weights.resampler_ln_post_bias = keepalive["ln_post_b"].data_ptr()
    weights.resampler_proj = keepalive["res_proj"].data_ptr()

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

    tgt_h = 14
    tgt_w = 14
    seq_len = tgt_h * tgt_w

    pixel_values = torch.randn((1, 3, patch, seq_len * patch), dtype=torch_dt)
    out_chain = torch.empty((resampler_meta.num_queries, resampler_meta.embed_dim), dtype=torch_dt)
    out_fused = torch.empty((resampler_meta.num_queries, resampler_meta.embed_dim), dtype=torch_dt)

    tmp = torch.empty((seq_len, d), dtype=torch_dt)
    model.infer_siglip_embeddings(model_handle, pixel_values.data_ptr(), seq_len, tgt_h, tgt_w, tmp.data_ptr())
    model.infer_siglip_encoder(model_handle, nlayer, tmp.data_ptr(), seq_len, tmp.data_ptr())
    model.infer_resampler(model_handle, tmp.data_ptr(), seq_len, tgt_h, tgt_w, out_chain.data_ptr())

    model.infer_vision_resampler(model_handle, pixel_values.data_ptr(), seq_len, tgt_h, tgt_w, out_fused.data_ptr())

    print(out_chain)
    print(out_fused)
    diff = (out_chain.float() - out_fused.float()).abs()
    print("vision->resampler chain smoke:")
    print("  dtype:", torch_dt)
    print("  out:", tuple(out_fused.shape))
    print("  max_abs:", diff.max().item())
    print("  mean_abs:", diff.mean().item())

    model.destroy_model(model_handle)


if __name__ == "__main__":
    main()
