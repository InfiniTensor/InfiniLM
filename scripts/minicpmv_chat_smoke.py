import argparse
import json
import os
from ctypes import POINTER, c_float, c_int, c_uint
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import safe_open

from libinfinicore_infer import (
    DataType,
    DeviceType,
    JiugeModel,
    KVCacheCStruct,
    KVCompressionConfigCStruct,
    MiniCPMVLanguageMetaCStruct,
    MiniCPMVMetaCStruct,
    MiniCPMVModel,
    MiniCPMVResamplerMetaCStruct,
    MiniCPMVVisionMetaCStruct,
    MiniCPMVWeightsCStruct,
    MiniCPMVSiglipLayerWeightsCStruct,
)


def _dtype_from_dt_logits(dt_logits: DataType):
    if dt_logits == DataType.INFINI_DTYPE_F32:
        return torch.float32
    if dt_logits == DataType.INFINI_DTYPE_BF16:
        return torch.bfloat16
    if dt_logits == DataType.INFINI_DTYPE_F16:
        return torch.float16
    raise ValueError(f"Unsupported dt_logits: {dt_logits}")


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


def _build_minicpmv_vision_model(model_dir: Path, torch_dt_logits, dt_logits: DataType):
    config = json.loads((model_dir / "config.json").read_text())
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]

    vision_cfg = config["vision_config"]
    patch = int(vision_cfg["patch_size"])
    d_v = int(vision_cfg["hidden_size"])
    nh_v = int(vision_cfg["num_attention_heads"])
    di_v = int(vision_cfg["intermediate_size"])
    nlayer = int(vision_cfg["num_hidden_layers"])

    language_meta = MiniCPMVLanguageMetaCStruct(
        dt_logits=dt_logits,
        nlayer=int(config["num_hidden_layers"]),
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

    # Vision weights
    keepalive["patch_w"] = _load_tensor(model_dir, weight_map, "vpm.embeddings.patch_embedding.weight").detach().to(dtype=torch_dt_logits).contiguous()
    keepalive["patch_b"] = _load_tensor(model_dir, weight_map, "vpm.embeddings.patch_embedding.bias").detach().to(dtype=torch_dt_logits).contiguous()
    keepalive["pos_emb"] = _load_tensor(model_dir, weight_map, "vpm.embeddings.position_embedding.weight").detach().to(dtype=torch_dt_logits).contiguous()

    layers = []
    for i in range(nlayer):
        lw, ka = _make_siglip_layer_struct(model_dir, weight_map, i, torch_dt_logits)
        layers.append(lw)
        for k, v in ka.items():
            keepalive[f"l{i}_{k}"] = v
    layers_arr = (MiniCPMVSiglipLayerWeightsCStruct * nlayer)(*layers)

    keepalive["post_ln_w"] = _load_tensor(model_dir, weight_map, "vpm.post_layernorm.weight").detach().to(dtype=torch_dt_logits).contiguous()
    keepalive["post_ln_b"] = _load_tensor(model_dir, weight_map, "vpm.post_layernorm.bias").detach().to(dtype=torch_dt_logits).contiguous()

    # Resampler weights (linear weights must be transposed to [in, out])
    def t(key: str) -> torch.Tensor:
        return _load_tensor(model_dir, weight_map, key).detach().to(dtype=torch_dt_logits).transpose(0, 1).contiguous()

    keepalive["res_kv_proj_w_t"] = t("resampler.kv_proj.weight")
    keepalive["res_in_w_t"] = t("resampler.attn.in_proj_weight")
    keepalive["res_out_w_t"] = t("resampler.attn.out_proj.weight")
    keepalive["res_in_b"] = _load_tensor(model_dir, weight_map, "resampler.attn.in_proj_bias").detach().to(dtype=torch_dt_logits).contiguous()
    keepalive["res_out_b"] = _load_tensor(model_dir, weight_map, "resampler.attn.out_proj.bias").detach().to(dtype=torch_dt_logits).contiguous()
    keepalive["res_query"] = _load_tensor(model_dir, weight_map, "resampler.query").detach().to(dtype=torch_dt_logits).contiguous()
    keepalive["res_proj"] = _load_tensor(model_dir, weight_map, "resampler.proj").detach().to(dtype=torch_dt_logits).contiguous()
    for name in ["ln_q", "ln_kv", "ln_post"]:
        keepalive[f"{name}_w"] = _load_tensor(model_dir, weight_map, f"resampler.{name}.weight").detach().to(dtype=torch_dt_logits).contiguous()
        keepalive[f"{name}_b"] = _load_tensor(model_dir, weight_map, f"resampler.{name}.bias").detach().to(dtype=torch_dt_logits).contiguous()

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

    # Language weights unused here
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

    # Keep ctypes objects alive: MiniCPMVModel stores pointers to `weights` and `vpm_layers`.
    keepalive["layers_arr"] = layers_arr
    keepalive["weights_struct"] = weights

    m = MiniCPMVModel()
    dev_ids = (c_int * 1)(0)
    handle = m.create_model(meta, weights, DeviceType.DEVICE_TYPE_CPU, 1, dev_ids)
    return m, handle, meta, keepalive


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--question", default="图片是什么？")
    ap.add_argument("--max-steps", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--kv-compress", action="store_true", help="Enable in-place KV cache compression after prefill.")
    ap.add_argument("--kv-compress-bin", default="", help="Path to compressor .bin weights.")
    ap.add_argument("--kv-compress-factor", type=int, default=5)
    ap.add_argument("--kv-compress-min-seq-len", type=int, default=2)
    ap.add_argument("--kv-compress-image-len", type=int, default=0, help="Prefix tokens treated as image KV (0 for Hybrid text-only).")
    args = ap.parse_args()
    debug = args.debug or os.environ.get("MINICPMV_DEBUG", "0") == "1"

    model_dir = Path(args.model_dir)

    # LLM (Jiuge) loader
    from jiuge import JiugeForCauslLM

    llm = JiugeForCauslLM(
        str(model_dir),
        device=DeviceType.DEVICE_TYPE_CPU,
        ndev=1,
        max_tokens=args.max_tokens,
    )

    # Build processor using the same tokenizer
    preproc_cfg = json.loads((model_dir / "preprocessor_config.json").read_text())
    from minicpmv_config.image_processing_minicpmv import MiniCPMVImageProcessor
    from minicpmv_config.processing_minicpmv import MiniCPMVProcessor

    image_processor = MiniCPMVImageProcessor(**preproc_cfg)
    processor = MiniCPMVProcessor(image_processor=image_processor, tokenizer=llm.tokenizer)

    # The vendored HF processor searches for the literal pattern `(<image>./</image>)`,
    # so we must include exactly one char + '/' inside the image tag.
    user_content = f"<image>./</image>\n{args.question}"
    prompt = llm.tokenizer.apply_chat_template(
        conversation=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_content},
        ],
        add_generation_prompt=True,
        tokenize=False,
    )

    img = Image.open(args.image).convert("RGB")
    batch = processor(text=prompt, images=[img], return_tensors="pt")

    input_ids = batch["input_ids"][0].to(dtype=torch.int64)
    attn = batch["attention_mask"][0].to(dtype=torch.bool)
    pad_left = int((~attn).sum().item())
    tokens = input_ids[pad_left:].to(dtype=torch.int32)

    bounds = batch["image_bound"][0].to(dtype=torch.int64)
    bounds = bounds - pad_left
    if bounds.shape[0] > 0:
        if debug:
            print("DEBUG pad_left:", pad_left)
            print("DEBUG tokens_len:", int(tokens.numel()))
            print("DEBUG bounds_all:", bounds.tolist())

    pixel_values_slices = batch["pixel_values"][0]
    tgt_sizes = batch["tgt_sizes"][0]

    # `image_bound` may include non-vision spans (e.g., <image_id>...</image_id>), which are not 64-token features.
    feature_len = int(preproc_cfg.get("image_feature_size", 64))
    bounds_all = bounds
    bounds = torch.stack([b for b in bounds_all if int((b[1] - b[0]).item()) == feature_len], dim=0)

    if bounds.shape[0] != bounds_all.shape[0]:
        if debug:
            print(
                f"INFO: filtered image_bound: total={bounds_all.shape[0]} feature_len={feature_len} kept={bounds.shape[0]}"
            )
            print("  image_bound_all (after left-pad adjust):", bounds_all.tolist())
            print("  image_bound_kept:", bounds.tolist())

    if len(pixel_values_slices) != bounds.shape[0]:
        if debug:
            print(f"WARNING: slice count mismatch: slices={len(pixel_values_slices)} bounds={bounds.shape[0]}")
        # Proceed by truncating to the common prefix (processor constructs placeholders in slice order).
        n = min(len(pixel_values_slices), int(bounds.shape[0]))
        bounds = bounds[:n]
        pixel_values_slices = pixel_values_slices[:n]
        tgt_sizes = tgt_sizes[:n]

    if len(pixel_values_slices) == 0:
        raise SystemExit("No image slices to run vision.")


    # Vision can be computed in f32 for numerical stability, then cast to LLM dtype for injection.
    llm_torch_dt = llm.meta.torch_dtype_logits
    llm_dt = llm.meta.dt_logits
    vision_force_f32 = os.environ.get("MINICPMV_VISION_FORCE_F32", "0") == "1"
    vision_torch_dt = torch.float32 if vision_force_f32 else llm_torch_dt
    vision_dt = DataType.INFINI_DTYPE_F32 if vision_force_f32 else llm_dt

    vision_model, vision_handle, vision_meta, vision_keepalive = _build_minicpmv_vision_model(
        model_dir, vision_torch_dt, vision_dt
    )

    # Compute per-slice vision embeddings [num_slices, 64, 3584]
    slice_embeds = []
    patch = int(preproc_cfg.get("patch_size", 14))
    for i, x in enumerate(pixel_values_slices):
        th, tw = int(tgt_sizes[i][0].item()), int(tgt_sizes[i][1].item())
        seq_len = th * tw
        x = x.to(dtype=vision_torch_dt).contiguous()
        packed = x.unsqueeze(0).contiguous()
        if packed.shape != (1, 3, patch, seq_len * patch):
            raise SystemExit(f"bad packed shape: {tuple(packed.shape)} for slice {i}")

        out = torch.empty(
            (vision_meta.resampler_meta.num_queries, vision_meta.resampler_meta.embed_dim),
            dtype=vision_torch_dt,
        )
        vision_model.infer_vision_resampler(vision_handle, packed.data_ptr(), seq_len, th, tw, out.data_ptr())
        if torch.isnan(out).any():
            nan_cnt = int(torch.isnan(out).sum().item())
            print(f"ERROR: vision out has NaN: slice={i} tgt_h={th} tgt_w={tw} nan_cnt={nan_cnt}")
            print(
                "  vision_out_abs_max/mean:",
                float(out.float().abs().max().item()),
                float(out.float().abs().mean().item()),
            )
            raise SystemExit("vision output contains NaNs")
        if out.dtype != llm_torch_dt:
            out = out.to(dtype=llm_torch_dt)
        slice_embeds.append(out.contiguous())

    # Flatten override positions and embeddings according to image_bound.
    override_pos_list = []
    override_embed_list = []
    for i in range(bounds.shape[0]):
        s = int(bounds[i][0].item())
        e = int(bounds[i][1].item())
        if e - s != vision_meta.resampler_meta.num_queries:
            raise SystemExit(f"unexpected bound length: {e-s} (expected {vision_meta.resampler_meta.num_queries})")
        override_pos_list.extend(list(range(s, e)))
        override_embed_list.append(slice_embeds[i])
    override_embeds = torch.cat(override_embed_list, dim=0).contiguous()
    if debug:
        print(
            "DEBUG override_embeds stats:",
            float(override_embeds.float().abs().max().item()),
            float(override_embeds.float().abs().mean().item()),
            override_embeds.dtype,
        )

    # Sanity: all override positions should correspond to `<unk>` tokens.
    unk_id = getattr(llm.tokenizer, "unk_token_id", None)
    if unk_id is not None:
        override_tok = tokens[torch.tensor(override_pos_list, dtype=torch.long)]
        uniq = torch.unique(override_tok).tolist()
        if len(uniq) != 1 or int(uniq[0]) != int(unk_id):
            if debug:
                print("WARNING: override positions are not all <unk> tokens.")
                print("  unk_id:", int(unk_id))
                print("  override_token_ids_unique:", [int(x) for x in uniq[:16]])

    override_pos = (c_uint * len(override_pos_list))(*override_pos_list)

    # Prefill with overrides
    ntok = int(tokens.numel())
    tokens_c = (c_uint * ntok)(*tokens.tolist())
    req_lens = (c_uint * 1)(ntok)
    req_pos = (c_uint * 1)(0)
    dev_ids = (c_int * 1)(0)

    kv = llm.jiuge_model.create_kv_cache(
        llm.meta.nlayer,
        llm.meta.dctx,
        llm.meta.nkvh,
        llm.meta.dh,
        llm.meta.dh,
        llm.meta.dt_logits,
        DeviceType.DEVICE_TYPE_CPU,
        dev_ids,
        1,
    )
    kv_caches = (POINTER(KVCacheCStruct) * 1)(kv)

    temperature = (c_float * 1)(1.0)
    topk = (c_uint * 1)(1)
    topp = (c_float * 1)(1.0)

    out = (c_uint * 1)()
    llm.jiuge_model.infer_batch_with_overrides(
        llm.model_instance,
        tokens_c,
        ntok,
        req_lens,
        1,
        req_pos,
        kv_caches,
        len(override_pos_list),
        override_pos,
        override_embeds.data_ptr(),
        temperature,
        topk,
        topp,
        out,
    )
    if debug:
        print("DEBUG prefill next_token:", int(out[0]))

    generated = [int(out[0])]
    rope_pos = ntok
    kv_pos = ntok
    eos_ids = set(llm.eos_token_id)

    if args.kv_compress:
        if not args.kv_compress_bin:
            raise SystemExit("--kv-compress requires --kv-compress-bin")
        cfg = KVCompressionConfigCStruct(
            enable=1,
            compression_factor=int(args.kv_compress_factor),
            min_seq_len=int(args.kv_compress_min_seq_len),
            image_kv_len=int(args.kv_compress_image_len),
            weight_path=args.kv_compress_bin.encode("utf-8"),
        )
        kv_pos = int(llm.jiuge_model.compress_kv_cache_inplace(kv, ntok, cfg))
        if debug:
            print("DEBUG kv_compress:", {"rope_pos": int(rope_pos), "kv_pos": int(kv_pos)})

    for _ in range(args.max_steps - 1):
        if generated[-1] in eos_ids:
            break
        req_lens = (c_uint * 1)(1)
        req_pos = (c_uint * 1)(rope_pos)
        kv_pos_c = (c_uint * 1)(kv_pos)
        tokens_c = (c_uint * 1)(generated[-1])
        if args.kv_compress:
            llm.jiuge_model.infer_batch_ex(
                llm.model_instance,
                tokens_c,
                1,
                req_lens,
                1,
                req_pos,
                kv_pos_c,
                kv_caches,
                temperature,
                topk,
                topp,
                out,
            )
        else:
            llm.jiuge_model.infer_batch(
                llm.model_instance,
                tokens_c,
                1,
                req_lens,
                1,
                req_pos,
                kv_caches,
                temperature,
                topk,
                topp,
                out,
            )
        generated.append(int(out[0]))
        rope_pos += 1
        kv_pos += 1

    if debug:
        print("DEBUG generated_ids:", generated)
    text = llm.tokenizer.decode(generated, skip_special_tokens=False)
    print(text)

    llm.jiuge_model.drop_kv_cache(kv)
    vision_model.destroy_model(vision_handle)
    llm.jiuge_model.destroy_model(llm.model_instance)


if __name__ == "__main__":
    main()
