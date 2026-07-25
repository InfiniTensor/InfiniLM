import ctypes
from typing import List
from tqdm import tqdm
import os
import sys
import time
import json
import torch
import transformers

from libinfinicore_infer import (
    DeepSeekOCRModel,
    DeepSeekOCRMetaCStruct,
    DeepSeekOCRWeightsCStruct,
    DataType,
    DeviceType,
)
from infer_task import InferTask, KVCache
from ctypes import POINTER, c_float, c_int, c_uint, c_void_p, byref

torch.set_default_device("cpu")


def load_specific_tensor(model_dir, tensor_name):
    """从safetensors模型加载特定tensor"""
    import safetensors

    index_file = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"Index file not found: {index_file}")

    with open(index_file, "r") as f:
        index = json.load(f)

    weight_map = index["weight_map"]
    if tensor_name not in weight_map:
        raise KeyError(f"{tensor_name} not found in index")

    filename = weight_map[tensor_name]
    tensor_file = os.path.join(model_dir, filename)

    with safetensors.safe_open(tensor_file, framework="pt", device="cpu") as f:
        tensor = f.get_tensor(tensor_name)
    return tensor


class DeepSeekOCRWeightsNaming:
    """DeepSeek-OCR权重命名规则"""

    def __init__(self, n_dense=1, n_sparse=11):
        self.n_dense = n_dense
        self.n_sparse = n_sparse

    def input_embd(self):
        return "model.embed_tokens.weight"

    def output_norm(self):
        return "model.norm.weight"

    def output_embd(self):
        return "lm_head.weight"

    # Attention layers
    def attn_norm(self, i):
        return f"model.layers.{i}.input_layernorm.weight"

    def attn_q_proj(self, i):
        return f"model.layers.{i}.self_attn.q_proj.weight"

    def attn_k_proj(self, i):
        return f"model.layers.{i}.self_attn.k_proj.weight"

    def attn_v_proj(self, i):
        return f"model.layers.{i}.self_attn.v_proj.weight"

    def attn_o_proj(self, i):
        return f"model.layers.{i}.self_attn.o_proj.weight"

    # FFN
    def ffn_norm(self, i):
        return f"model.layers.{i}.post_attention_layernorm.weight"

    # Dense MLP (layer 0)
    def dense_gate_proj(self, i):
        assert i < self.n_dense
        return f"model.layers.{i}.mlp.gate_proj.weight"

    def dense_up_proj(self, i):
        assert i < self.n_dense
        return f"model.layers.{i}.mlp.up_proj.weight"

    def dense_down_proj(self, i):
        assert i < self.n_dense
        return f"model.layers.{i}.mlp.down_proj.weight"

    # MoE layers (layer 1-11)
    def moe_gate_weight(self, i):
        assert i >= self.n_dense
        return f"model.layers.{i}.mlp.gate.weight"

    def moe_gate_bias(self, i):
        assert i >= self.n_dense
        return f"model.layers.{i}.mlp.gate.e_score_correction_bias"

    def moe_shared_gate_proj(self, i):
        assert i >= self.n_dense
        return f"model.layers.{i}.mlp.shared_experts.gate_proj.weight"

    def moe_shared_up_proj(self, i):
        assert i >= self.n_dense
        return f"model.layers.{i}.mlp.shared_experts.up_proj.weight"

    def moe_shared_down_proj(self, i):
        assert i >= self.n_dense
        return f"model.layers.{i}.mlp.shared_experts.down_proj.weight"

    def moe_experts_gate_proj(self, i, e):
        assert i >= self.n_dense
        return f"model.layers.{i}.mlp.experts.{e}.gate_proj.weight"

    def moe_experts_up_proj(self, i, e):
        assert i >= self.n_dense
        return f"model.layers.{i}.mlp.experts.{e}.up_proj.weight"

    def moe_experts_down_proj(self, i, e):
        assert i >= self.n_dense
        return f"model.layers.{i}.mlp.experts.{e}.down_proj.weight"

    # Vision Encoder weights
    def projector(self):
        return "model.projector.layers.weight"

    def image_newline(self):
        return "model.image_newline"

    def view_seperator(self):
        return "model.view_seperator"

    # SAM ViT-B weights (12 layers)
    def sam_patch_embed_weight(self):
        return "model.sam_model.patch_embed.proj.weight"

    def sam_patch_embed_bias(self):
        return "model.sam_model.patch_embed.proj.bias"

    def sam_block_norm1(self, layer):
        return f"model.sam_model.blocks.{layer}.norm1.weight"

    def sam_block_attn_qkv(self, layer):
        return f"model.sam_model.blocks.{layer}.attn.qkv.weight"

    def sam_block_attn_proj(self, layer):
        return f"model.sam_model.blocks.{layer}.attn.proj.weight"

    def sam_block_norm2(self, layer):
        return f"model.sam_model.blocks.{layer}.norm2.weight"

    def sam_block_mlp_fc1(self, layer):
        return f"model.sam_model.blocks.{layer}.mlp.lin1.weight"

    def sam_block_mlp_fc2(self, layer):
        return f"model.sam_model.blocks.{layer}.mlp.lin2.weight"

    def sam_neck_conv1(self):
        return "model.sam_model.neck.0.weight"

    def sam_neck_ln1(self):
        return "model.sam_model.neck.1.weight"

    def sam_neck_conv2(self):
        return "model.sam_model.neck.2.weight"

    def sam_neck_ln2(self):
        return "model.sam_model.neck.3.weight"

    # CLIP-L weights (24 layers)
    def clip_patch_embed_weight(self):
        return "model.vision_model.embeddings.patch_embedding.weight"

    def clip_patch_embed_bias(self):
        return "model.vision_model.embeddings.patch_embedding.bias"

    def clip_position_embed(self):
        return "model.vision_model.embeddings.position_embedding.weight"

    def clip_pre_layernorm(self):
        return "model.vision_model.pre_layrnorm.weight"

    def clip_block_ln1(self, layer):
        return f"model.vision_model.encoder.layers.{layer}.layer_norm1.weight"

    def clip_block_attn_qkv(self, layer):
        return f"model.vision_model.encoder.layers.{layer}.self_attn.qkv.weight"

    def clip_block_attn_proj(self, layer):
        return f"model.vision_model.encoder.layers.{layer}.self_attn.out_proj.weight"

    def clip_block_ln2(self, layer):
        return f"model.vision_model.encoder.layers.{layer}.layer_norm2.weight"

    def clip_block_mlp_fc1(self, layer):
        return f"model.vision_model.encoder.layers.{layer}.mlp.fc1.weight"

    def clip_block_mlp_fc2(self, layer):
        return f"model.vision_model.encoder.layers.{layer}.mlp.fc2.weight"


class DeepSeekOCRMeta(DeepSeekOCRMetaCStruct):
    def __init__(self, config, dtype=torch.bfloat16, max_tokens=None):
        if dtype == torch.float16:
            dt_ = DataType.INFINI_DTYPE_F16
        elif dtype == torch.bfloat16:
            dt_ = DataType.INFINI_DTYPE_BF16
        else:
            dt_ = DataType.INFINI_DTYPE_BF16

        n_dense = config.get("first_k_dense_replace", 1)
        n_sparse = config["num_hidden_layers"] - n_dense

        super().__init__(
            dt_logits=dt_,
            dt_norm=dt_,
            n_dense_layer=n_dense,
            n_sparse_layer=n_sparse,
            d=config["hidden_size"],
            nh=config["num_attention_heads"],
            nkvh=config.get("num_key_value_heads", config["num_attention_heads"]),
            dh=config["hidden_size"] // config["num_attention_heads"],
            di_dense=config.get("intermediate_size", 6848),
            di_moe=config.get("moe_intermediate_size", 896),
            di_shared=config.get("shared_expert_intermediate_size", 1792),
            nexperts=config.get("n_routed_experts", 64),
            kexperts=config.get("num_experts_per_tok", 6),
            routed_scale=config.get("routed_scaling_factor", 1.0),
            dctx=config["max_position_embeddings"] if max_tokens is None else max_tokens,
            dvoc=config["vocab_size"],
            epsilon=config.get("rms_norm_eps", 1e-6),
            theta=config.get("rope_theta", 10000.0),
            end_token=config.get("eos_token_id", 1),
        )
        self.torch_dtype_logits = dtype


def load_deepseek_ocr_weights(
    meta: DeepSeekOCRMeta, weights, model_path: str, ndev: int
):
    """加载DeepSeek-OCR所有权重"""
    from ctypes import cast

    names = DeepSeekOCRWeightsNaming()
    nlayer = meta.n_dense_layer + meta.n_sparse_layer

    # 基础embeddings
    input_embd = load_specific_tensor(model_path, names.input_embd()).to(meta.torch_dtype_logits)
    weights.input_embd = input_embd.data_ptr()

    output_norm = load_specific_tensor(model_path, names.output_norm()).to(meta.torch_dtype_logits)
    weights.output_norm = output_norm.data_ptr()

    output_embd = load_specific_tensor(model_path, names.output_embd()).to(meta.torch_dtype_logits)
    weights.output_embd = output_embd.data_ptr()

    # Attention & FFN norm (所有层)
    attn_norm_ptrs = (c_void_p * nlayer)()
    attn_q_ptrs = (c_void_p * nlayer)()
    attn_k_ptrs = (c_void_p * nlayer)()
    attn_v_ptrs = (c_void_p * nlayer)()
    attn_o_ptrs = (c_void_p * nlayer)()
    ffn_norm_ptrs = (c_void_p * nlayer)()

    layer_tensors = []  # 保持引用

    for i in tqdm(range(nlayer), desc="Loading layers"):
        attn_norm = load_specific_tensor(model_path, names.attn_norm(i)).to(meta.torch_dtype_logits)
        attn_norm_ptrs[i] = attn_norm.data_ptr()
        layer_tensors.append(attn_norm)

        attn_q = load_specific_tensor(model_path, names.attn_q_proj(i)).to(meta.torch_dtype_logits)
        attn_q_ptrs[i] = attn_q.data_ptr()
        layer_tensors.append(attn_q)

        attn_k = load_specific_tensor(model_path, names.attn_k_proj(i)).to(meta.torch_dtype_logits)
        attn_k_ptrs[i] = attn_k.data_ptr()
        layer_tensors.append(attn_k)

        attn_v = load_specific_tensor(model_path, names.attn_v_proj(i)).to(meta.torch_dtype_logits)
        attn_v_ptrs[i] = attn_v.data_ptr()
        layer_tensors.append(attn_v)

        attn_o = load_specific_tensor(model_path, names.attn_o_proj(i)).to(meta.torch_dtype_logits)
        attn_o_ptrs[i] = attn_o.data_ptr()
        layer_tensors.append(attn_o)

        ffn_norm = load_specific_tensor(model_path, names.ffn_norm(i)).to(meta.torch_dtype_logits)
        ffn_norm_ptrs[i] = ffn_norm.data_ptr()
        layer_tensors.append(ffn_norm)

    weights.attn_norm = cast(attn_norm_ptrs, POINTER(c_void_p))
    weights.attn_q = cast(attn_q_ptrs, POINTER(c_void_p))
    weights.attn_k = cast(attn_k_ptrs, POINTER(c_void_p))
    weights.attn_v = cast(attn_v_ptrs, POINTER(c_void_p))
    weights.attn_o = cast(attn_o_ptrs, POINTER(c_void_p))
    weights.ffn_norm = cast(ffn_norm_ptrs, POINTER(c_void_p))

    # Dense MLP (第0层)
    dense_gate = load_specific_tensor(model_path, names.dense_gate_proj(0)).to(meta.torch_dtype_logits)
    weights.dense_gate = dense_gate.data_ptr()

    dense_up = load_specific_tensor(model_path, names.dense_up_proj(0)).to(meta.torch_dtype_logits)
    weights.dense_up = dense_up.data_ptr()

    dense_down = load_specific_tensor(model_path, names.dense_down_proj(0)).to(meta.torch_dtype_logits)
    weights.dense_down = dense_down.data_ptr()

    # MoE (第1-11层)
    n_sparse = meta.n_sparse_layer
    moe_gate_weight_ptrs = (c_void_p * n_sparse)()
    moe_gate_bias_ptrs = (c_void_p * n_sparse)()
    moe_shared_gate_ptrs = (c_void_p * n_sparse)()
    moe_shared_up_ptrs = (c_void_p * n_sparse)()
    moe_shared_down_ptrs = (c_void_p * n_sparse)()

    moe_tensors = []
    for i in tqdm(range(1, nlayer), desc="Loading MoE layers"):
        moe_idx = i - 1

        gate_w = load_specific_tensor(model_path, names.moe_gate_weight(i)).to(meta.torch_dtype_logits)
        moe_gate_weight_ptrs[moe_idx] = gate_w.data_ptr()
        moe_tensors.append(gate_w)

        gate_b = load_specific_tensor(model_path, names.moe_gate_bias(i)).to(meta.torch_dtype_logits)
        moe_gate_bias_ptrs[moe_idx] = gate_b.data_ptr()
        moe_tensors.append(gate_b)

        shared_g = load_specific_tensor(model_path, names.moe_shared_gate_proj(i)).to(meta.torch_dtype_logits)
        moe_shared_gate_ptrs[moe_idx] = shared_g.data_ptr()
        moe_tensors.append(shared_g)

        shared_u = load_specific_tensor(model_path, names.moe_shared_up_proj(i)).to(meta.torch_dtype_logits)
        moe_shared_up_ptrs[moe_idx] = shared_u.data_ptr()
        moe_tensors.append(shared_u)

        shared_d = load_specific_tensor(model_path, names.moe_shared_down_proj(i)).to(meta.torch_dtype_logits)
        moe_shared_down_ptrs[moe_idx] = shared_d.data_ptr()
        moe_tensors.append(shared_d)

    weights.moe_gate_weight = cast(moe_gate_weight_ptrs, POINTER(c_void_p))
    weights.moe_gate_bias = cast(moe_gate_bias_ptrs, POINTER(c_void_p))
    weights.moe_shared_gate = cast(moe_shared_gate_ptrs, POINTER(c_void_p))
    weights.moe_shared_up = cast(moe_shared_up_ptrs, POINTER(c_void_p))
    weights.moe_shared_down = cast(moe_shared_down_ptrs, POINTER(c_void_p))

    # Routed experts
    nexperts = meta.nexperts
    expert_gate_ptrs_per_layer = []
    expert_up_ptrs_per_layer = []
    expert_down_ptrs_per_layer = []
    expert_tensors = []

    for moe_layer in tqdm(range(1, nlayer), desc="Loading MoE experts"):
        expert_gate_ptrs = (c_void_p * nexperts)()
        expert_up_ptrs = (c_void_p * nexperts)()
        expert_down_ptrs = (c_void_p * nexperts)()

        for e in range(nexperts):
            gate = load_specific_tensor(model_path, names.moe_experts_gate_proj(moe_layer, e)).to(meta.torch_dtype_logits)
            expert_gate_ptrs[e] = gate.data_ptr()
            expert_tensors.append(gate)

            up = load_specific_tensor(model_path, names.moe_experts_up_proj(moe_layer, e)).to(meta.torch_dtype_logits)
            expert_up_ptrs[e] = up.data_ptr()
            expert_tensors.append(up)

            down = load_specific_tensor(model_path, names.moe_experts_down_proj(moe_layer, e)).to(meta.torch_dtype_logits)
            expert_down_ptrs[e] = down.data_ptr()
            expert_tensors.append(down)

        expert_gate_ptrs_per_layer.append(cast(expert_gate_ptrs, POINTER(c_void_p)))
        expert_up_ptrs_per_layer.append(cast(expert_up_ptrs, POINTER(c_void_p)))
        expert_down_ptrs_per_layer.append(cast(expert_down_ptrs, POINTER(c_void_p)))

    weights.moe_experts_gate = cast((POINTER(c_void_p) * n_sparse)(*expert_gate_ptrs_per_layer), POINTER(POINTER(c_void_p)))
    weights.moe_experts_up = cast((POINTER(c_void_p) * n_sparse)(*expert_up_ptrs_per_layer), POINTER(POINTER(c_void_p)))
    weights.moe_experts_down = cast((POINTER(c_void_p) * n_sparse)(*expert_down_ptrs_per_layer), POINTER(POINTER(c_void_p)))

    # 视觉编码器权重
    vision_tensors = []

    # SAM ViT-B (12 layers)
    sam_patch_embed = load_specific_tensor(model_path, names.sam_patch_embed_weight()).to(meta.torch_dtype_logits)
    weights.sam_patch_embed = sam_patch_embed.data_ptr()
    vision_tensors.append(sam_patch_embed)

    sam_patch_embed_bias = load_specific_tensor(model_path, names.sam_patch_embed_bias()).to(meta.torch_dtype_logits)
    weights.sam_patch_embed_bias = sam_patch_embed_bias.data_ptr()
    vision_tensors.append(sam_patch_embed_bias)

    sam_block_norm1_ptrs = (c_void_p * 12)()
    sam_block_attn_qkv_ptrs = (c_void_p * 12)()
    sam_block_attn_proj_ptrs = (c_void_p * 12)()
    sam_block_norm2_ptrs = (c_void_p * 12)()
    sam_block_mlp_fc1_ptrs = (c_void_p * 12)()
    sam_block_mlp_fc2_ptrs = (c_void_p * 12)()

    for layer in tqdm(range(12), desc="Loading SAM blocks"):
        norm1 = load_specific_tensor(model_path, names.sam_block_norm1(layer)).to(meta.torch_dtype_logits)
        sam_block_norm1_ptrs[layer] = norm1.data_ptr()
        vision_tensors.append(norm1)

        qkv = load_specific_tensor(model_path, names.sam_block_attn_qkv(layer)).to(meta.torch_dtype_logits)
        sam_block_attn_qkv_ptrs[layer] = qkv.data_ptr()
        vision_tensors.append(qkv)

        proj = load_specific_tensor(model_path, names.sam_block_attn_proj(layer)).to(meta.torch_dtype_logits)
        sam_block_attn_proj_ptrs[layer] = proj.data_ptr()
        vision_tensors.append(proj)

        norm2 = load_specific_tensor(model_path, names.sam_block_norm2(layer)).to(meta.torch_dtype_logits)
        sam_block_norm2_ptrs[layer] = norm2.data_ptr()
        vision_tensors.append(norm2)

        fc1 = load_specific_tensor(model_path, names.sam_block_mlp_fc1(layer)).to(meta.torch_dtype_logits)
        sam_block_mlp_fc1_ptrs[layer] = fc1.data_ptr()
        vision_tensors.append(fc1)

        fc2 = load_specific_tensor(model_path, names.sam_block_mlp_fc2(layer)).to(meta.torch_dtype_logits)
        sam_block_mlp_fc2_ptrs[layer] = fc2.data_ptr()
        vision_tensors.append(fc2)

    weights.sam_block_norm1 = cast(sam_block_norm1_ptrs, POINTER(c_void_p))
    weights.sam_block_attn_qkv = cast(sam_block_attn_qkv_ptrs, POINTER(c_void_p))
    weights.sam_block_attn_proj = cast(sam_block_attn_proj_ptrs, POINTER(c_void_p))
    weights.sam_block_norm2 = cast(sam_block_norm2_ptrs, POINTER(c_void_p))
    weights.sam_block_mlp_fc1 = cast(sam_block_mlp_fc1_ptrs, POINTER(c_void_p))
    weights.sam_block_mlp_fc2 = cast(sam_block_mlp_fc2_ptrs, POINTER(c_void_p))

    sam_neck_conv1 = load_specific_tensor(model_path, names.sam_neck_conv1()).to(meta.torch_dtype_logits)
    weights.sam_neck_conv1 = sam_neck_conv1.data_ptr()
    vision_tensors.append(sam_neck_conv1)

    sam_neck_ln1 = load_specific_tensor(model_path, names.sam_neck_ln1()).to(meta.torch_dtype_logits)
    weights.sam_neck_ln1 = sam_neck_ln1.data_ptr()
    vision_tensors.append(sam_neck_ln1)

    sam_neck_conv2 = load_specific_tensor(model_path, names.sam_neck_conv2()).to(meta.torch_dtype_logits)
    weights.sam_neck_conv2 = sam_neck_conv2.data_ptr()
    vision_tensors.append(sam_neck_conv2)

    sam_neck_ln2 = load_specific_tensor(model_path, names.sam_neck_ln2()).to(meta.torch_dtype_logits)
    weights.sam_neck_ln2 = sam_neck_ln2.data_ptr()
    vision_tensors.append(sam_neck_ln2)

    # CLIP-L (24 layers)
    clip_patch_embed = load_specific_tensor(model_path, names.clip_patch_embed_weight()).to(meta.torch_dtype_logits)
    weights.clip_patch_embed = clip_patch_embed.data_ptr()
    vision_tensors.append(clip_patch_embed)

    clip_patch_embed_bias = load_specific_tensor(model_path, names.clip_patch_embed_bias()).to(meta.torch_dtype_logits)
    weights.clip_patch_embed_bias = clip_patch_embed_bias.data_ptr()
    vision_tensors.append(clip_patch_embed_bias)

    clip_position_embed = load_specific_tensor(model_path, names.clip_position_embed()).to(meta.torch_dtype_logits)
    weights.clip_position_embed = clip_position_embed.data_ptr()
    vision_tensors.append(clip_position_embed)

    clip_pre_layernorm = load_specific_tensor(model_path, names.clip_pre_layernorm()).to(meta.torch_dtype_logits)
    weights.clip_pre_layernorm = clip_pre_layernorm.data_ptr()
    vision_tensors.append(clip_pre_layernorm)

    clip_block_ln1_ptrs = (c_void_p * 24)()
    clip_block_attn_qkv_ptrs = (c_void_p * 24)()
    clip_block_attn_proj_ptrs = (c_void_p * 24)()
    clip_block_ln2_ptrs = (c_void_p * 24)()
    clip_block_mlp_fc1_ptrs = (c_void_p * 24)()
    clip_block_mlp_fc2_ptrs = (c_void_p * 24)()

    for layer in tqdm(range(24), desc="Loading CLIP blocks"):
        ln1 = load_specific_tensor(model_path, names.clip_block_ln1(layer)).to(meta.torch_dtype_logits)
        clip_block_ln1_ptrs[layer] = ln1.data_ptr()
        vision_tensors.append(ln1)

        qkv = load_specific_tensor(model_path, names.clip_block_attn_qkv(layer)).to(meta.torch_dtype_logits)
        clip_block_attn_qkv_ptrs[layer] = qkv.data_ptr()
        vision_tensors.append(qkv)

        proj = load_specific_tensor(model_path, names.clip_block_attn_proj(layer)).to(meta.torch_dtype_logits)
        clip_block_attn_proj_ptrs[layer] = proj.data_ptr()
        vision_tensors.append(proj)

        ln2 = load_specific_tensor(model_path, names.clip_block_ln2(layer)).to(meta.torch_dtype_logits)
        clip_block_ln2_ptrs[layer] = ln2.data_ptr()
        vision_tensors.append(ln2)

        fc1 = load_specific_tensor(model_path, names.clip_block_mlp_fc1(layer)).to(meta.torch_dtype_logits)
        clip_block_mlp_fc1_ptrs[layer] = fc1.data_ptr()
        vision_tensors.append(fc1)

        fc2 = load_specific_tensor(model_path, names.clip_block_mlp_fc2(layer)).to(meta.torch_dtype_logits)
        clip_block_mlp_fc2_ptrs[layer] = fc2.data_ptr()
        vision_tensors.append(fc2)

    weights.clip_block_ln1 = cast(clip_block_ln1_ptrs, POINTER(c_void_p))
    weights.clip_block_attn_qkv = cast(clip_block_attn_qkv_ptrs, POINTER(c_void_p))
    weights.clip_block_attn_proj = cast(clip_block_attn_proj_ptrs, POINTER(c_void_p))
    weights.clip_block_ln2 = cast(clip_block_ln2_ptrs, POINTER(c_void_p))
    weights.clip_block_mlp_fc1 = cast(clip_block_mlp_fc1_ptrs, POINTER(c_void_p))
    weights.clip_block_mlp_fc2 = cast(clip_block_mlp_fc2_ptrs, POINTER(c_void_p))

    # Projector
    projector = load_specific_tensor(model_path, names.projector()).to(meta.torch_dtype_logits)
    weights.projector = projector.data_ptr()
    vision_tensors.append(projector)

    image_newline = load_specific_tensor(model_path, names.image_newline()).to(meta.torch_dtype_logits)
    weights.image_newline = image_newline.data_ptr()
    vision_tensors.append(image_newline)

    view_seperator = load_specific_tensor(model_path, names.view_seperator()).to(meta.torch_dtype_logits)
    weights.view_seperator = view_seperator.data_ptr()
    vision_tensors.append(view_seperator)

    return layer_tensors + moe_tensors + expert_tensors + vision_tensors + [
        input_embd, output_norm, output_embd, dense_gate, dense_up, dense_down
    ]


class DeepSeekOCRBatchedTask:
    def __init__(self, tasks: List[InferTask]):
        from libinfinicore_infer import KVCacheCStruct

        self.tasks = tasks
        self.nreq = len(tasks)

        token_lists = [t.tokens for t in tasks]
        self.req_lens_list = [len(toks) for toks in token_lists]
        self.req_pos_list = [t.pos for t in tasks]
        self.kv_cache_ptrs = [t.kvcache().data() for t in tasks]
        self.temperaturas_list = [t.temperature for t in tasks]
        self.topks_list = [t.topk for t in tasks]
        self.topps_list = [t.topp for t in tasks]

        flat_tokens = [tok for toks in token_lists for tok in toks]
        self.ntok = len(flat_tokens)

        self.tokens = (c_uint * self.ntok)(*flat_tokens)
        self.req_lens = (c_uint * self.nreq)(*self.req_lens_list)
        self.req_pos = (c_uint * self.nreq)(*self.req_pos_list)
        self.kv_caches = (POINTER(KVCacheCStruct) * self.nreq)(*self.kv_cache_ptrs)
        self.temperaturas = (c_float * self.nreq)(*self.temperaturas_list)
        self.topks = (c_uint * self.nreq)(*self.topks_list)
        self.topps = (c_float * self.nreq)(*self.topps_list)

    def input_args(self):
        return (
            self.tokens,
            self.ntok,
            self.req_lens,
            self.nreq,
            self.req_pos,
            self.kv_caches,
            self.temperaturas,
            self.topks,
            self.topps,
        )


class DeepSeekOCRForCausalLM:
    def __init__(
        self,
        model_dir_path,
        device=DeviceType.DEVICE_TYPE_CPU,
        ndev=1,
        max_tokens=None,
    ):
        with open(os.path.join(model_dir_path, "config.json"), "r") as f:
            config = json.load(f)
            self.config = config

        eos_token_id = self.config["eos_token_id"]
        self.eos_token_id = (
            [eos_token_id] if type(eos_token_id) == int else eos_token_id
        )

        print(f"Loading model from: {model_dir_path}")

        # 创建meta
        self.meta = DeepSeekOCRMeta(config, max_tokens=max_tokens, dtype=torch.bfloat16)

        # 加载tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_dir_path, trust_remote_code=True
        )

        # 创建C++模型并加载权重
        print(f"Creating model on {ndev} devices...")
        load_start_time = time.time()
        dev_ids = (c_int * ndev)(*[i for i in range(ndev)])

        self.model_instance = DeepSeekOCRModel()

        # 创建权重结构
        from ctypes import Structure, POINTER
        weights = DeepSeekOCRWeightsCStruct()
        weights.n_dense_layer = self.meta.n_dense_layer
        weights.n_sparse_layer = self.meta.n_sparse_layer
        weights.dt_norm = self.meta.dt_norm
        weights.dt_mat = self.meta.dt_logits
        weights.transpose_linear_weights = 1  # PyTorch format is transposed

        # 加载所有权重
        print("Loading model weights...")
        self.weight_tensors = load_deepseek_ocr_weights(self.meta, weights, model_dir_path, ndev)

        # 创建模型
        self.model_ptr = self.model_instance.create_model(
            byref(self.meta),
            byref(weights),
            device,
            ndev,
            dev_ids,
        )

        load_end_time = time.time()
        print(f"Time used: {load_end_time - load_start_time:.3f}s")


    def max_context_len(self):
        return self.meta.dctx

    def create_kv_cache(self):
        nlayer = self.meta.n_dense_layer + self.meta.n_sparse_layer
        return self.model_instance.create_kv_cache(
            nlayer,
            self.meta.dctx,
            self.meta.nkvh,
            self.meta.dh,
            self.meta.dh,
            self.meta.dt_logits,
            self.model_ptr.contents.resources[0].device if hasattr(self.model_ptr.contents, 'resources') else DeviceType.DEVICE_TYPE_CPU,
            (c_int * 1)(0),
            1,
        )

    def drop_kv_cache(self, kv_cache):
        self.model_instance.drop_kv_cache(kv_cache)


    def generate(self, input_content, max_steps, image_path=None,
                 topp_=1.0, topk_=1, temperature_=1.0):
        """生成文本(支持多模态输入)"""
        # 构建prompt
        if image_path:
            input_content = f"<image>\n{input_content}"

        # Tokenize
        tokens = self.tokenizer.encode(input_content)

        # 创建推理任务
        infer_task = InferTask(
            0, tokens, self.max_context_len(),
            temperature_, topk_, topp_, self.eos_token_id
        )
        infer_task.bind_kvcache(KVCache(self))

        print(input_content, end="", flush=True)

        steps = 0
        total_time = 0
        output_content = ""

        # 生成循环
        for step_i in range(max_steps):
            start_time = time.time()

            # 调用推理
            output_tokens = self.batch_infer_one_round([infer_task])

            end_time = time.time()
            steps += 1

            # 解码
            output_str = self.tokenizer.decode(output_tokens[0])
            output_content += output_str
            print(output_str, end="", flush=True)

            # 检查结束
            if output_tokens[0] in self.eos_token_id:
                break

            infer_task.next(output_tokens[0])

            if step_i > 0:
                total_time += end_time - start_time

        print("\n")
        avg_time = total_time * 1000 / max(steps - 1, 1)
        print(f"Time per step: {avg_time:.3f}ms")

        infer_task._kv_cache.drop(self)
        return output_content, avg_time

    def batch_infer_one_round(self, tasks: List[InferTask]):
        output = (c_uint * len(tasks))()
        batch_inputs = DeepSeekOCRBatchedTask(tasks)
        self.model_instance.infer_batch(
            self.model_ptr, *(batch_inputs.input_args()), output
        )
        return list(output)

    def destroy_model_instance(self):
        self.model_instance.destroy_model(self.model_ptr)
        print("Model destroyed")


def test():
    if len(sys.argv) < 3:
        print(
            "Usage: python deepseek_ocr.py [--cpu | --nvidia | --cambricon | --ascend] <path/to/model_dir> [n_device]"
        )
        sys.exit(1)

    model_path = sys.argv[2]
    device_type = DeviceType.DEVICE_TYPE_CPU

    if sys.argv[1] == "--cpu":
        device_type = DeviceType.DEVICE_TYPE_CPU
    elif sys.argv[1] == "--nvidia":
        device_type = DeviceType.DEVICE_TYPE_NVIDIA
    elif sys.argv[1] == "--cambricon":
        device_type = DeviceType.DEVICE_TYPE_CAMBRICON
    elif sys.argv[1] == "--ascend":
        device_type = DeviceType.DEVICE_TYPE_ASCEND
    else:
        print(
            "Usage: python deepseek_ocr.py [--cpu | --nvidia | --cambricon | --ascend] <path/to/model_dir> [n_device]"
        )
        sys.exit(1)

    ndev = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    model = DeepSeekOCRForCausalLM(model_path, device_type, ndev, max_tokens=2048)

    # 测试纯文本生成
    output, avg_time = model.generate("北京是中国的首都吗？", 50)
    print(f"\nAverage time per step: {avg_time:.3f}ms")

    model.destroy_model_instance()


if __name__ == "__main__":
    test()
