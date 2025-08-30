import ctypes
from ctypes import (
    POINTER, Structure, c_size_t, c_float, c_int, c_int32, c_uint, c_void_p, c_bool
)
import os
import sys

# ===================================================================
# 1. Generic Definitions
# ===================================================================
# ... (This part remains unchanged) ...
class DeviceType(c_int32):
    DEVICE_TYPE_CPU = 0
    DEVICE_TYPE_NVIDIA = 1
    DEVICE_TYPE_CAMBRICON = 2
    DEVICE_TYPE_ASCEND = 3
    DEVICE_TYPE_METAX = 4
    DEVICE_TYPE_MOORE = 5
    DEVICE_TYPE_ILUVATAR = 6

class DataType(ctypes.c_int):
    INFINI_DTYPE_INVALID = 0
    INFINI_DTYPE_BYTE = 1
    INFINI_DTYPE_BOOL = 2
    INFINI_DTYPE_I8 = 3
    INFINI_DTYPE_I16 = 4
    INFINI_DTYPE_I32 = 5
    INFINI_DTYPE_I64 = 6
    INFINI_DTYPE_U8 = 7
    INFINI_DTYPE_U16 = 8
    INFINI_DTYPE_U32 = 9
    INFINI_DTYPE_U64 = 10
    INFINI_DTYPE_F8 = 11
    INFINI_DTYPE_F16 = 12
    INFINI_DTYPE_F32 = 13
    INFINI_DTYPE_F64 = 14
    INFINI_DTYPE_C16 = 15
    INFINI_DTYPE_C32 = 16
    INFINI_DTYPE_C64 = 17
    INFINI_DTYPE_C128 = 18
    INFINI_DTYPE_BF16 = 19

class KVCacheCStruct(ctypes.Structure):
    pass

# ===================================================================
# 2. Dense Model Definitions
# ===================================================================
# ... (This part remains unchanged) ...
class QwenMetaCStruct(Structure):
    _fields_ = [
        ("dt_logits", DataType), ("nlayer", c_size_t), ("d", c_size_t),
        ("nh", c_size_t), ("nkvh", c_size_t), ("dh", c_size_t),
        ("di", c_size_t), ("dctx", c_size_t), ("dvoc", c_size_t),
        ("epsilon", c_float), ("theta", c_float), ("end_token", c_uint),
    ]

class QwenWeightsCStruct(Structure):
    _fields_ = [
        ("nlayer", c_size_t), ("dt_norm", DataType), ("dt_mat", DataType),
        ("transpose_linear_weights", c_int), ("input_embd", c_void_p),
        ("output_norm", c_void_p), ("output_embd", c_void_p),
        ("attn_norm", POINTER(c_void_p)), ("attn_qkv", POINTER(c_void_p)),
        ("attn_qkv_b", POINTER(c_void_p)), ("attn_q_norm", POINTER(c_void_p)),
        ("attn_k_norm", POINTER(c_void_p)), ("attn_o", POINTER(c_void_p)),
        ("ffn_norm", POINTER(c_void_p)), ("ffn_gate_up", POINTER(c_void_p)),
        ("ffn_down", POINTER(c_void_p)),
    ]

class QwenModelCStruct(ctypes.Structure):
    pass

# ===================================================================
# 3. MoE Model Definitions
# ===================================================================
# ... (This part remains unchanged) ...
class QwenMoeMetaCStruct(Structure):
    _fields_ = [
        ("dt_logits", DataType), ("nlayer", c_size_t), ("d", c_size_t),
        ("nh", c_size_t), ("nkvh", c_size_t), ("dh", c_size_t),
        ("di", c_size_t), ("dctx", c_size_t), ("dvoc", c_size_t),
        ("epsilon", c_float), ("theta", c_float), ("end_token", c_uint),
        ("num_experts", c_size_t), ("num_experts_per_tok", c_size_t),
        ("moe_intermediate_size", c_size_t), ("norm_topk_prob", c_int),
    ]

class QwenMoeWeightsCStruct(Structure):
    _fields_ = [
        ("nlayer", c_size_t), ("dt_norm", DataType), ("dt_mat", DataType),
        ("transpose_linear_weights", c_int), ("input_embd", c_void_p),
        ("output_norm", c_void_p), ("output_embd", c_void_p),
        ("attn_norm", POINTER(c_void_p)), ("attn_qkv", POINTER(c_void_p)),
        ("attn_qkv_b", POINTER(c_void_p)), ("attn_q_norm", POINTER(c_void_p)),
        ("attn_k_norm", POINTER(c_void_p)), ("attn_o", POINTER(c_void_p)),
        ("ffn_norm", POINTER(c_void_p)), ("moe_gate", POINTER(c_void_p)),
        ("moe_experts_gate_up", POINTER(c_void_p)),
        ("moe_experts_down", POINTER(c_void_p)),
    ]

class QwenMoeModelCStruct(ctypes.Structure):
    pass

# ===================================================================
# 4. Library Loading and Function Definitions
# ===================================================================

# --- 仅加载库文件，但不初始化任何函数 ---
try:
    lib_path = os.path.join(
        os.environ.get("INFINI_ROOT", "."), "lib", "libinfinicore_infer.so"
    )
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"Library not found at {lib_path}")
    LIB = ctypes.CDLL(lib_path)
    print("Successfully located C++ library.", file=sys.stderr)
except (FileNotFoundError, OSError) as e:
    print(f"FATAL: Could not load C++ library: {e}", file=sys.stderr)
    LIB = None

# --- 按需初始化函数 ---

def initialize_dense_apis():
    """按需加载并返回 Dense 模型的 API 函数"""
    if not LIB: return (None,) * 6
    try:
        LIB.createQwenModel.restype = POINTER(QwenModelCStruct)
        LIB.createQwenModel.argtypes = [ POINTER(QwenMetaCStruct), POINTER(QwenWeightsCStruct), DeviceType, c_int, POINTER(c_int) ]
        LIB.destroyQwenModel.argtypes = [POINTER(QwenModelCStruct)]
        LIB.createKVCache.restype = POINTER(KVCacheCStruct)
        LIB.createKVCache.argtypes = [POINTER(QwenModelCStruct)]
        LIB.dropKVCache.argtypes = [POINTER(QwenModelCStruct), POINTER(KVCacheCStruct)]
        LIB.inferBatch.argtypes = [ POINTER(QwenModelCStruct), POINTER(c_uint), c_uint, POINTER(c_uint), c_uint, POINTER(c_uint), POINTER(POINTER(KVCacheCStruct)), POINTER(c_float), POINTER(c_uint), POINTER(c_float), POINTER(c_uint) ]
        LIB.forwardBatch.argtypes = [ POINTER(QwenModelCStruct), POINTER(c_uint), c_uint, POINTER(c_uint), c_uint, POINTER(c_uint), POINTER(POINTER(KVCacheCStruct)), c_void_p ]
        print("Successfully loaded REAL Dense Model functions.", file=sys.stderr)
        return LIB.createQwenModel, LIB.destroyQwenModel, LIB.createKVCache, LIB.dropKVCache, LIB.inferBatch, LIB.forwardBatch
    except AttributeError as e:
        print(f"ERROR: Could not load Dense Model functions: {e}", file=sys.stderr)
        return (None,) * 6

def initialize_moe_apis():
    """按需加载并返回 MoE 模型的 API 函数（如果失败则返回模拟函数）"""
    if not LIB: # 如果库文件本身就没找到，直接返回模拟函数
        return mock_all_apis()

    try:
        LIB.createQwenMoeModel.restype = POINTER(QwenMoeModelCStruct)
        LIB.createQwenMoeModel.argtypes = [ POINTER(QwenMoeMetaCStruct), POINTER(QwenMoeWeightsCStruct), DeviceType, c_int, POINTER(c_int) ]
        LIB.destroyQwenMoeModel.argtypes = [POINTER(QwenMoeModelCStruct)]
        LIB.createQwenMoeKVCache.restype = POINTER(KVCacheCStruct)
        LIB.createQwenMoeKVCache.argtypes = [POINTER(QwenMoeModelCStruct)]
        LIB.dropQwenMoeKVCache.argtypes = [POINTER(QwenMoeModelCStruct), POINTER(KVCacheCStruct)]
        LIB.inferQwenMoeBatch.argtypes = [ POINTER(QwenMoeModelCStruct), POINTER(c_uint), c_uint, POINTER(c_uint), c_uint, POINTER(c_uint), POINTER(POINTER(KVCacheCStruct)), POINTER(c_float), POINTER(c_uint), POINTER(c_float), POINTER(c_uint) ]
        LIB.forwardQwenMoeBatch.argtypes = [ POINTER(QwenMoeModelCStruct), POINTER(c_uint), c_uint, POINTER(c_uint), c_uint, POINTER(c_uint), POINTER(POINTER(KVCacheCStruct)), c_void_p ]
        print("Successfully loaded REAL MoE Model functions.", file=sys.stderr)
        return LIB.createQwenMoeModel, LIB.destroyQwenMoeModel, LIB.createQwenMoeKVCache, LIB.dropQwenMoeKVCache, LIB.inferQwenMoeBatch, LIB.forwardQwenMoeBatch
    except AttributeError as e:
        print(f"WARNING: Could not load MoE Model functions due to '{e}'. Creating mocks.", file=sys.stderr)
        return mock_all_apis()

def mock_all_apis():
    """返回一套完整的模拟函数"""
    def mock_create_model(*args):
        print(f"MOCK: create_model function called. Returning dummy model.", file=sys.stderr)
        return POINTER(QwenMoeModelCStruct)()
    def mock_create_kv_cache(*args):
        print("MOCK: create_kv_cache called. Returning dummy cache.", file=sys.stderr)
        return POINTER(KVCacheCStruct)()
    def mock_void_function(*args):
        print(f"MOCK: A void function (like destroy or infer) was called.", file=sys.stderr)
        pass
    return mock_create_model, mock_void_function, mock_create_kv_cache, mock_void_function, mock_void_function, mock_void_function
