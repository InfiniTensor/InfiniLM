import ctypes
from ctypes import c_size_t, c_uint, c_int, c_float, c_void_p, c_bool, POINTER
import torch
from .base import DataType, DeviceType
import ctypes
import os
from typing import List



def find_name_in_state_dict(name_list: List[str], state_dict: dict):
    retname = None
    for name in name_list:
        if name in state_dict:
            retname = name
            break
    return retname

class MLPCStruct(ctypes.Structure):
    _fields_ = [
        ("_gate_up_proj_weight", c_void_p),
        ("_down_proj_weight", c_void_p),
    ]

    def __init__(self, ilayer: int, di, ndev, d,
                 torch_dt_mat, transpose_weight,
                 state_dict: dict,
                 gate_proj=None, up_proj=None, down_proj=None):
        # transpose_weight 默认为True

        ### gate_up
        self.gate_up_tensor = torch.concat(self.gate_up_slices(ilayer, di, ndev, state_dict, gate_proj=gate_proj, up_proj=up_proj)).to(torch_dt_mat)

        if not transpose_weight:
            self.gate_up_tensor = self.gate_up_tensors.reshape(ndev, 2 * di // ndev, d).transpose(1, 2).contiguous()
        setattr(self, "_gate_up_proj_weight", self.gate_up_tensor.data_ptr())

        ### down
        if down_proj is None:
            down_proj = find_name_in_state_dict([f"model.layers.{ilayer}.mlp.down_proj.weight", f"layers.{ilayer}.mlp.down_proj.weight"], state_dict)
        if transpose_weight:
            self.ffn_down_tensor = state_dict[down_proj].to(torch_dt_mat).reshape([d, ndev, di // ndev]).transpose(0, 1).contiguous()
        else:
            self.ffn_down_tensor = state_dict[down_proj].transpose(0, 1).to(torch_dt_mat).contiguous()

        setattr(self, "_down_proj_weight", self.ffn_down_tensor.data_ptr())

    def gate_up_slices(self, ilayer: int, di, ndev, state_dict: dict,
                       gate_proj=None, up_proj=None):
        if gate_proj is None:
            gate_proj = find_name_in_state_dict([f"model.layers.{ilayer}.mlp.gate_proj.weight", f"layers.{ilayer}.mlp.gate_proj.weight"], state_dict)
        if up_proj is None:
            up_proj = find_name_in_state_dict([f"model.layers.{ilayer}.mlp.up_proj.weight", f"layers.{ilayer}.mlp.up_proj.weight"], state_dict)

        _result = []
        _di = di // ndev
        for _idev in range(ndev):
            _start = _idev * _di
            _end = (_idev + 1) * _di
            _result.append(state_dict[gate_proj][_start:_end, :])
            _result.append(state_dict[up_proj][_start:_end, :])

        return _result


class AttentionCStruct(ctypes.Structure):
    _fields_ = [
        ("_qkv_proj_weight", c_void_p),
        ("_qkv_proj_bias", c_void_p),
        ("_qk_norm_weight", c_void_p),
        ("_o_proj_weight", c_void_p),
    ]

    def __init__(self, ilayer: int, nh, nkvh, d, dh, ndev,
                 torch_dt_mat, torch_dt_logits, torch_dt_norm,
                 transpose_weight,
                 state_dict: dict):
        ###
        self.qkv_tensor = torch.concat(self.qkv_slices(ilayer, nh, nkvh, d, dh, ndev, state_dict)).to(torch_dt_mat)
        if not transpose_weight:
            self.qkv_tensor = self.qkv_tensor.reshape(ndev, (nh + 2 * nkvh) // ndev * dh, d).transpose(1, 2).contiguous()
        setattr(self, "_qkv_proj_weight", self.qkv_tensor.data_ptr())

        ###
        self.qkv_b_tensor = None
        attn_q_b = f"model.layers.{ilayer}.self_attn.q_proj.bias"
        if attn_q_b in state_dict:
            self.qkv_b_tensor = torch.concat(self.qkv_b_slices(ilayer, nh, nkvh, d, dh, ndev, state_dict)).to(torch_dt_logits)
            setattr(self, "_qkv_proj_bias", self.qkv_b_tensor.data_ptr())

        ###
        self.qk_norm_tensor = None
        attn_q_norm = find_name_in_state_dict([f"model.layers.{ilayer}.self_attn.q_norm.weight", f"layers.{ilayer}.self_attn.q_norm.weight"], state_dict)
        if attn_q_norm in state_dict:
            attn_q_norm = find_name_in_state_dict([f"model.layers.{ilayer}.self_attn.q_norm.weight", f"layers.{ilayer}.self_attn.q_norm.weight"], state_dict)
            attn_k_norm = find_name_in_state_dict([f"model.layers.{ilayer}.self_attn.k_norm.weight", f"layers.{ilayer}.self_attn.k_norm.weight"], state_dict)

            q_norm = state_dict[attn_q_norm].reshape([2, dh // 2]).transpose(1, 0)
            k_norm = state_dict[attn_k_norm].reshape([2, dh // 2]).transpose(1, 0)
            self.qk_norm_tensor = torch.concat([q_norm, k_norm]).to(torch_dt_norm)
            setattr(self, "_qk_norm_weight", self.qk_norm_tensor.data_ptr())

        ###
        attn_o = find_name_in_state_dict([f"model.layers.{ilayer}.self_attn.o_proj.weight", f"layers.{ilayer}.self_attn.o_proj.weight"], state_dict)
        if transpose_weight:
            self.attn_o_tensor = state_dict[attn_o].to(torch_dt_mat).reshape([d, ndev, nh // ndev * dh]).transpose(0, 1).contiguous()
        else:
            self.attn_o_tensor = state_dict[attn_o].transpose(0, 1).to(torch_dt_mat).contiguous()
        setattr(self, "_o_proj_weight", self.attn_o_tensor.data_ptr())

    def qkv_b_slices(self, ilayer, nh, nkvh, d, dh, ndev, state_dict):
        attn_q_b = f"model.layers.{ilayer}.self_attn.q_proj.bias"
        attn_k_b = f"model.layers.{ilayer}.self_attn.k_proj.bias"
        attn_v_b = f"model.layers.{ilayer}.self_attn.v_proj.bias"

        _QB = state_dict[attn_q_b].reshape([nh, 2, dh // 2]).transpose(1, 2)
        _KB = state_dict[attn_k_b].reshape([nkvh, 2, dh // 2]).transpose(1, 2)
        _VB = state_dict[attn_v_b].reshape([nkvh, dh // 2, 2])

        _result = []
        _nh = nh // ndev
        _nkvh = nkvh // ndev
        for _idev in range(ndev):
            _result.append(_QB[_idev * _nh: (_idev + 1) * _nh, :, :].flatten())
            _result.append(_KB[_idev * _nkvh: (_idev + 1) * _nkvh, :, :].flatten())
            _result.append(_VB[_idev * _nkvh: (_idev + 1) * _nkvh, :, :].flatten())
        return _result

    def qkv_slices(self, ilayer: int, nh, nkvh, d, dh, ndev, state_dict):
        attn_q = find_name_in_state_dict([f"model.layers.{ilayer}.self_attn.q_proj.weight", f"layers.{ilayer}.self_attn.q_proj.weight"], state_dict)
        attn_k = find_name_in_state_dict([f"model.layers.{ilayer}.self_attn.k_proj.weight", f"layers.{ilayer}.self_attn.k_proj.weight"], state_dict)
        attn_v = find_name_in_state_dict([f"model.layers.{ilayer}.self_attn.v_proj.weight", f"layers.{ilayer}.self_attn.v_proj.weight"], state_dict)

        _Q = state_dict[attn_q].reshape([nh, 2, dh // 2, d]).transpose(1, 2)
        _K = state_dict[attn_k].reshape([nkvh, 2, dh // 2, d]).transpose(1, 2)
        _V = state_dict[attn_v].reshape([nkvh, dh // 2, 2, d])

        _result = []
        _nh = nh // ndev
        _nkvh = nkvh // ndev
        for _idev in range(ndev):
            _result.append(_Q[_idev * _nh: (_idev + 1) * _nh, :, :, :])
            _result.append(_K[_idev * _nkvh: (_idev + 1) * _nkvh, :, :, :])
            _result.append(_V[_idev * _nkvh: (_idev + 1) * _nkvh, :, :])
        return _result



class MoEMetaCStruct(ctypes.Structure):
    _fields_ = [
        ("dt_logits", DataType),
        ("nlayer", c_size_t),
        ("d", c_size_t),
        ("nh", c_size_t),
        ("nkvh", c_size_t),
        ("dh", c_size_t),
        ("di", c_size_t),
        ("dctx", c_size_t),
        ("dvoc", c_size_t),
        ("epsilon", c_float),
        ("theta", c_float),
        ("end_token", c_uint),
        #
        ("_moe_intermediate_size", c_size_t),
        ("_shared_expert_intermediate_size", c_size_t),
        ("_num_experts", c_size_t),
        ("_num_experts_per_tok", c_size_t),
        ("_norm_topk_prob", c_bool),
    ]


class SparseMLPCStruct(ctypes.Structure):
    _fields_ = [
        ("_shared_expert_num", c_size_t),
        ("_num_experts", c_size_t),
        ("_shared_expert_gate_weight", c_void_p),
        ("_gate_weight", c_void_p),
        ("_shared_expert", MLPCStruct),
        ("_experts", POINTER(MLPCStruct)),
    ]

    def __init__(self, ilayer: int, num_experts, ndev, d,
                 torch_dt_mat, transpose_weight,
                 _moe_intermediate_size, _shared_expert_intermediate_size, _num_experts_per_tok, _norm_topk_prob,
                 state_dict: dict):

        setattr(self, "_num_experts", num_experts)

        # shared_expert
        shared_expert_gate = f"model.layers.{ilayer}.mlp.shared_expert_gate.weight"
        if shared_expert_gate in state_dict:

            self.shared_expert_gate_tensor = state_dict[shared_expert_gate].to(torch_dt_mat)
            if transpose_weight:
                self.shared_expert_gate_tensor = self.shared_expert_gate_tensor.transpose(0, 1).contiguous()
            setattr(self, "_shared_expert_gate_weight", self.shared_expert_gate_tensor.data_ptr())
            setattr(self, "_shared_expert_num", 1)

            ## shared_expert
            gate_proj = f"model.layers.{ilayer}.mlp.shared_expert.gate_proj.weight"
            up_proj = f"model.layers.{ilayer}.mlp.shared_expert.up_proj.weight"
            down_proj = f"model.layers.{ilayer}.mlp.shared_expert.down_proj.weight"
            self.shared_expert_mlp = MLPCStruct(ilayer, _shared_expert_intermediate_size, ndev, d,
                                                torch_dt_mat, transpose_weight,
                                                state_dict,
                                                gate_proj=gate_proj, up_proj=up_proj, down_proj=down_proj)
            setattr(self, "_shared_expert", self.shared_expert_mlp)
        else:
            setattr(self, "_shared_expert_num", 0)

        ## experts
        experts_gate = f"model.layers.{ilayer}.mlp.gate.weight"
        self.experts_gate_tensor = state_dict[experts_gate].to(torch_dt_mat)
        if transpose_weight:
            self.experts_gate_tensor = self.experts_gate_tensor.transpose(0, 1).contiguous()
        setattr(self, "_gate_weight", self.experts_gate_tensor.data_ptr())

        self.experts_mlp = []
        for i in range(num_experts):
            gate_proj = f"model.layers.{ilayer}.mlp.experts.{i}.gate_proj.weight"
            up_proj = f"model.layers.{ilayer}.mlp.experts.{i}.up_proj.weight"
            down_proj = f"model.layers.{ilayer}.mlp.experts.{i}.down_proj.weight"
            self.experts_mlp.append(
                MLPCStruct(ilayer, _moe_intermediate_size, ndev, d,
                           torch_dt_mat, transpose_weight,
                           state_dict,
                           gate_proj=gate_proj, up_proj=up_proj, down_proj=down_proj)
            )

        self.experts_mlp_array = (MLPCStruct * num_experts)(*self.experts_mlp)
        setattr(self, "_experts", self.experts_mlp_array)


# Define the Decoder Layer struct
class DecoderLayerCStruct(ctypes.Structure):
    _fields_ = [
        ("_ilayer", c_int),
        ("_post_attention_layernorm_weight", c_void_p),
        ("_input_layernorm_weight", c_void_p),
        ("_self_attn", AttentionCStruct),
        ("_mlp", SparseMLPCStruct),
    ]

    def __init__(self, ilayer: int, num_experts, nh, nkvh, d, di, dh, ndev,
                 torch_dt_mat, torch_dt_logits, torch_dt_norm,
                 transpose_weight,
                 _moe_intermediate_size, _shared_expert_intermediate_size, _num_experts_per_tok, _norm_topk_prob,
                 state_dict: dict):
        setattr(self, "_ilayer", ilayer)

        attn_norm = f"model.layers.{ilayer}.input_layernorm.weight"
        self.attn_norm_tensor = state_dict[attn_norm].to(torch_dt_norm)
        setattr(self, "_input_layernorm_weight", self.attn_norm_tensor.data_ptr())

        ffn_norm = f"model.layers.{ilayer}.post_attention_layernorm.weight"
        self.mlp_norm_tensor = state_dict[ffn_norm].to(torch_dt_norm)
        setattr(self, "_post_attention_layernorm_weight", self.mlp_norm_tensor.data_ptr())

        self.self_attn = AttentionCStruct(ilayer, nh, nkvh, d, dh, ndev, torch_dt_mat, torch_dt_logits, torch_dt_norm, transpose_weight, state_dict)
        setattr(self, "_self_attn", self.self_attn)

        self.mlp = SparseMLPCStruct(ilayer, num_experts, ndev, d, torch_dt_mat, transpose_weight,
                                    _moe_intermediate_size, _shared_expert_intermediate_size, _num_experts_per_tok, _norm_topk_prob,
                                    state_dict)
        setattr(self, "_mlp", self.mlp)


# Define the QwenWeights struct
class WeightsCStruct(ctypes.Structure):
    _fields_ = [
        ("_nlayer", c_size_t),
        ("_dt_norm", DataType),
        ("_dt_mat", DataType),
        ("_transpose_linear_weights", c_int),
        ###
        ("_embed_tokens_weight", c_void_p),
        ("_norm_weight", c_void_p),
        ("_lm_head_weight", c_void_p),
        ###
        ("_layers", POINTER(DecoderLayerCStruct)),
    ]

    def __init__(self, nlayer, num_experts, nh, nkvh, d, di, dh, ndev,
                 torch_dt_mat, torch_dt_logits, torch_dt_norm,
                 transpose_weight,
                 _moe_intermediate_size, _shared_expert_intermediate_size, _num_experts_per_tok, _norm_topk_prob,
                 state_dict: dict):
        ###
        setattr(self, "_nlayer", nlayer)
        setattr(self, "_transpose_linear_weights", 1 if transpose_weight else 0)

        if torch_dt_mat == torch.float16:
            setattr(self, "_dt_mat", DataType.INFINI_DTYPE_F16)
        elif torch_dt_mat == torch.float32:
            setattr(self, "_dt_mat", DataType.INFINI_DTYPE_F32)
        elif torch_dt_mat == torch.bfloat16:
            setattr(self, "_dt_mat", DataType.INFINI_DTYPE_BF16)
        else:
            raise ValueError("Unsupported proj weight data type")

        if torch_dt_norm == torch.float16:
            setattr(self, "_dt_norm", DataType.INFINI_DTYPE_F16)
        elif torch_dt_norm == torch.float32:
            setattr(self, "_dt_norm", DataType.INFINI_DTYPE_F32)
        elif torch_dt_norm == torch.bfloat16:
            setattr(self, "_dt_norm", DataType.INFINI_DTYPE_BF16)
        else:
            raise ValueError("Unsupported norm weight data type")

        ###
        input_embd = "model.embed_tokens.weight"
        output_norm = "model.norm.weight"
        output_embd = "lm_head.weight"

        input_embd_naming = input_embd if input_embd in state_dict else output_embd
        self.input_embd_tensor = state_dict[input_embd_naming].to(torch_dt_logits)

        setattr(self, "_embed_tokens_weight", self.input_embd_tensor.data_ptr())

        output_embd_naming = output_embd if output_embd in state_dict else input_embd
        self.output_embd_tensor = state_dict[output_embd_naming].to(torch_dt_mat)  # 这里把输入数据强制类型转换了 ？？ 使用的不是 bf16 了
        if not transpose_weight:
            self.output_embd_tensor = self.output_embd_tensor.transpose(0, 1).contiguous()
        setattr(self, "_lm_head_weight", self.output_embd_tensor.data_ptr())

        self.output_norm_tensor = state_dict[output_norm].to(torch_dt_norm)
        setattr(self, "_norm_weight", self.output_norm_tensor.data_ptr())

        ###
        self.layers = []
        for ilayer in range(nlayer):
            self.layers.append(
                DecoderLayerCStruct(ilayer, num_experts, nh, nkvh, d, di, dh, ndev,
                                    torch_dt_mat, torch_dt_logits, torch_dt_norm,
                                    transpose_weight,
                                    _moe_intermediate_size, _shared_expert_intermediate_size, _num_experts_per_tok, _norm_topk_prob,
                                    state_dict)
            )

        self.layers_array = (DecoderLayerCStruct * nlayer)(*self.layers)
        setattr(self, "_layers", self.layers_array)


class ModelCSruct(ctypes.Structure):
    pass


class KVCacheCStruct(ctypes.Structure):
    pass


def __open_library__():
    lib_path = os.path.join(
        os.environ.get("INFINI_ROOT"), "lib", "libinfinicore_infer.so"
    )

    lib = ctypes.CDLL(lib_path)
    lib.Qwen3MoEcreateModel.restype = POINTER(ModelCSruct)
    lib.Qwen3MoEcreateModel.argtypes = [
        POINTER(MoEMetaCStruct),  # JiugeMeta const *
        POINTER(WeightsCStruct),  # JiugeWeights const *
        DeviceType,  # DeviceType
        c_int,  # int ndev
        POINTER(c_int),  # int const *dev_ids
    ]
    lib.Qwen3MoEdestroyModel.argtypes = [POINTER(ModelCSruct)]
    lib.Qwen3MoEcreateKVCache.argtypes = [POINTER(ModelCSruct)]
    lib.Qwen3MoEcreateKVCache.restype = POINTER(KVCacheCStruct)
    lib.Qwen3MoEdropKVCache.argtypes = [POINTER(ModelCSruct), POINTER(KVCacheCStruct)]
    lib.Qwen3MoEinferBatch.restype = None
    lib.Qwen3MoEinferBatch.argtypes = [
        POINTER(ModelCSruct),  # struct JiugeModel const *
        POINTER(c_uint),  # unsigned int const *tokens
        c_uint,  # unsigned int ntok
        POINTER(c_uint),  # unsigned int const *req_lens
        c_uint,  # unsigned int nreq
        POINTER(c_uint),  # unsigned int const *req_pos
        POINTER(POINTER(KVCacheCStruct)),  # struct KVCache **kv_caches
        POINTER(c_float),  # float temperature
        POINTER(c_uint),  # unsigned int topk
        POINTER(c_float),  # float topp
        POINTER(c_uint),  # unsigned int *output
    ]
    lib.Qwen3MoEforwardBatch.restype = None
    lib.Qwen3MoEforwardBatch.argtypes = [
        POINTER(ModelCSruct),  # struct JiugeModel const *
        POINTER(c_uint),  # unsigned int const *tokens
        c_uint,  # unsigned int ntok
        POINTER(c_uint),  # unsigned int const *req_lens
        c_uint,  # unsigned int nreq
        POINTER(c_uint),  # unsigned int const *req_pos
        POINTER(POINTER(KVCacheCStruct)),  # struct KVCache **kv_caches
        c_void_p,  # void *logits
    ]

    return lib


LIB = __open_library__()

create_model = LIB.Qwen3MoEcreateModel
destroy_model = LIB.Qwen3MoEdestroyModel
create_kv_cache = LIB.Qwen3MoEcreateKVCache
drop_kv_cache = LIB.Qwen3MoEdropKVCache
infer_batch = LIB.Qwen3MoEinferBatch
forward_batch = LIB.Qwen3MoEforwardBatch
