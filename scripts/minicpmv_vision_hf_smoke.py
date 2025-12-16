import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to an input image")
    ap.add_argument("--slice-idx", type=int, default=0)
    ap.add_argument("--all-slices", action="store_true")
    ap.add_argument("--max-slices", type=int, default=None)
    ap.add_argument("--max-slice-nums", type=int, default=None)
    args = ap.parse_args()

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

    from minicpmv_config.image_processing_minicpmv import MiniCPMVImageProcessor

    preproc_cfg = json.loads(((model_dir / "preprocessor_config.json").read_text()))
    ip = MiniCPMVImageProcessor(**preproc_cfg)

    img = Image.open(args.image).convert("RGB")
    batch = ip.preprocess(img, do_pad=True, max_slice_nums=args.max_slice_nums, return_tensors="pt")
    pixel_values = batch["pixel_values"][0]
    tgt_sizes = batch["tgt_sizes"][0]

    if not pixel_values:
        raise SystemExit("No slices produced by image processor.")

    if args.all_slices:
        slice_indices = list(range(len(pixel_values)))
        if args.max_slices is not None:
            slice_indices = slice_indices[: int(args.max_slices)]
    else:
        slice_idx = int(args.slice_idx)
        if slice_idx < 0 or slice_idx >= len(pixel_values):
            raise SystemExit(f"slice_idx out of range: {slice_idx} (num_slices={len(pixel_values)})")
        slice_indices = [slice_idx]

    patch = int(preproc_cfg.get("patch_size", 14))

    # ---------- Build C++ model (vision+resampler weights only) ----------
    vision_cfg = config["vision_config"]
    nlayer = int(vision_cfg["num_hidden_layers"])
    d_v = int(vision_cfg["hidden_size"])
    nh_v = int(vision_cfg["num_attention_heads"])
    di_v = int(vision_cfg["intermediate_size"])

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
        vision_embed_dim=d_v,
        vision_num_layers=nlayer,
        vision_num_heads=nh_v,
        vision_intermediate_size=di_v,
        vision_layer_norm_eps=1e-6,
        vision_image_size=int(vision_cfg["image_size"]),
        vision_num_positions=4900,
    )
    resampler_meta = MiniCPMVResamplerMetaCStruct(
        num_queries=int(config["query_num"]),
        embed_dim=int(config["hidden_size"]),
        num_heads=int(config["num_attention_heads"]),
        kv_dim=d_v,
        layer_norm_eps=1e-6,
        max_patches_h=70,
        max_patches_w=70,
    )
    meta = MiniCPMVMetaCStruct(
        vision_meta=vision_meta, resampler_meta=resampler_meta, language_meta=language_meta
    )

    keepalive = {}

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

    # ---------- Torch reference (vpm + resampler) ----------
    from minicpmv_config.modeling_navit_siglip import SiglipVisionConfig, SiglipVisionTransformer
    from minicpmv_config.resampler import Resampler

    vcfg = SiglipVisionConfig(
        hidden_size=d_v,
        intermediate_size=di_v,
        num_hidden_layers=nlayer,
        num_attention_heads=nh_v,
        num_channels=3,
        image_size=int(vision_cfg["image_size"]),
        patch_size=patch,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
    )
    vcfg._attn_implementation = "eager"
    vpm = SiglipVisionTransformer(vcfg).to(dtype=torch_dt).eval()

    # Load vpm weights by stripping the "vpm." prefix.
    vpm_sd = {}
    for k in weight_map.keys():
        if k.startswith("vpm."):
            vpm_sd[k[len("vpm.") :]] = _load_tensor(model_dir, weight_map, k).to(dtype=torch_dt)
    vpm.load_state_dict(vpm_sd, strict=True)

    resampler = Resampler(
        num_queries=int(config["query_num"]),
        embed_dim=int(config["hidden_size"]),
        num_heads=int(config["hidden_size"] // 128),
        kv_dim=d_v,
        adaptive=True,
    ).to(dtype=torch_dt).eval()
    res_sd = {}
    for k in weight_map.keys():
        if k.startswith("resampler."):
            res_sd[k[len("resampler.") :]] = _load_tensor(model_dir, weight_map, k).to(dtype=torch_dt)
    resampler.load_state_dict(res_sd, strict=True)

    overall_max = 0.0
    overall_sum = 0.0
    overall_n = 0

    for slice_idx in slice_indices:
        x = pixel_values[slice_idx].to(dtype=torch_dt).contiguous()  # [3, patch, L]
        th, tw = int(tgt_sizes[slice_idx][0].item()), int(tgt_sizes[slice_idx][1].item())
        seq_len = th * tw
        assert x.shape[0] == 3 and x.shape[1] == patch
        assert x.shape[2] == seq_len * patch
        packed = x.unsqueeze(0).contiguous()  # [1, 3, patch, L]

        out_cpp = torch.empty((resampler_meta.num_queries, resampler_meta.embed_dim), dtype=torch_dt)
        model.infer_vision_resampler(model_handle, packed.data_ptr(), seq_len, th, tw, out_cpp.data_ptr())

        patch_attn_mask = torch.ones((1, 1, seq_len), dtype=torch.bool)
        tgt = torch.tensor([[th, tw]], dtype=torch.int32)
        with torch.no_grad():
            hs = vpm(packed, patch_attention_mask=patch_attn_mask, tgt_sizes=tgt).last_hidden_state
            out_ref = resampler(hs, tgt)[0]

        diff = (out_cpp.float() - out_ref.float()).abs()
        max_abs = float(diff.max().item())
        mean_abs = float(diff.mean().item())

        overall_max = max(overall_max, max_abs)
        overall_sum += float(diff.sum().item())
        overall_n += diff.numel()

        print("vision hf smoke:")
        print("  slice_idx:", slice_idx)
        print("  tgt_h:", th, "tgt_w:", tw, "seq_len:", seq_len)
        print("  dtype:", torch_dt)
        print("  out:", tuple(out_cpp.shape))
        print("  max_abs:", max_abs)
        print("  mean_abs:", mean_abs)

    if len(slice_indices) > 1:
        print("vision hf smoke summary:")
        print("  slices:", len(slice_indices))
        print("  overall_max_abs:", overall_max)
        print("  overall_mean_abs:", overall_sum / max(1, overall_n))

    model.destroy_model(model_handle)


if __name__ == "__main__":
    main()
