import ctypes
from ctypes import c_size_t, c_uint, c_int, c_float, c_void_p, c_bool, POINTER
import os


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


class DeviceType(ctypes.c_int):
    DEVICE_TYPE_CPU = 0
    DEVICE_TYPE_NVIDIA = 1
    DEVICE_TYPE_CAMBRICON = 2
    DEVICE_TYPE_ASCEND = 3
    DEVICE_TYPE_METAX = 4
    DEVICE_TYPE_MOORE = 5
    DEVICE_TYPE_ILUVATAR = 6


class Qwen3MetaCStruct(ctypes.Structure):
    _fields_ = [
        # 数据类型
        ("dt_logits", DataType),        # infiniDtype_t dt_logits
        
        # 网络规模
        ("nlayer", c_size_t),          # size_t nlayer
        ("d", c_size_t),               # size_t d
        ("nh", c_size_t),              # size_t nh
        ("nkvh", c_size_t),            # size_t nkvh
        ("dh", c_size_t),              # size_t dh
        ("di", c_size_t),              # size_t di
        ("dctx", c_size_t),            # size_t dctx
        ("dvoc", c_size_t),            # size_t dvoc
        
        # 归一化与RoPE
        ("epsilon", c_float),          # float epsilon
        ("theta", c_float),            # float theta
        
        # 额外字段
        ("bos_token", c_uint),         # uint32_t bos_token
        ("end_token", c_uint),         # uint32_t end_token
        ("attn_dropout", c_float),     # float attn_dropout
        ("tie_embd", c_bool),          # bool tie_embd
    ]


class Qwen3WeightsCStruct(ctypes.Structure):
    _fields_ = [
        # 元信息
        ("nlayer", c_size_t),          # size_t nlayer
        ("dt_norm", DataType),         # infiniDtype_t dt_norm
        ("dt_mat", DataType),          # infiniDtype_t dt_mat
        ("transpose_linear_weights", c_int),  # int transpose_linear_weights
        
        # 全局共享权重
        ("input_embd", c_void_p),      # const void *input_embd
        ("output_embd", c_void_p),     # const void *output_embd
        ("output_norm", c_void_p),     # const void *output_norm
        
        # 逐层权重数组 (长度为nlayer)
        ("attn_norm", POINTER(c_void_p)),      # const void **attn_norm
        ("attn_q_norm", POINTER(c_void_p)),    # const void **attn_q_norm
        ("attn_k_norm", POINTER(c_void_p)),    # const void **attn_k_norm
        
        # QKV投影 - 分开存放
        ("attn_q_proj", POINTER(c_void_p)),    # const void **attn_q_proj
        ("attn_k_proj", POINTER(c_void_p)),    # const void **attn_k_proj
        ("attn_v_proj", POINTER(c_void_p)),    # const void **attn_v_proj
        ("attn_o_proj", POINTER(c_void_p)),    # const void **attn_o_proj
        
        # MLP层
        ("mlp_norm", POINTER(c_void_p)),       # const void **mlp_norm
        ("mlp_gate_proj", POINTER(c_void_p)),  # const void **mlp_gate_proj
        ("mlp_up_proj", POINTER(c_void_p)),    # const void **mlp_up_proj
        ("mlp_down_proj", POINTER(c_void_p)),  # const void **mlp_down_proj
    ]


class Qwen3ModelCStruct(ctypes.Structure):
    pass


class KVCacheCStruct(ctypes.Structure):
    pass


def __open_library__():
    infini_root = os.environ.get("INFINI_ROOT")
    if infini_root is None:
        raise EnvironmentError(
            "INFINI_ROOT environment variable not set. "
            "Please set it to the InfiniCore installation directory or run 'xmake install' first."
        )
    print("p",infini_root)
    lib_path = os.path.join(infini_root, "lib", "libinfinicore_infer.so")
    if not os.path.exists(lib_path):
        raise FileNotFoundError(
            f"Library not found at {lib_path}. "
            "Please compile the library with 'xmake' and install with 'xmake install' first."
        )
    lib = ctypes.CDLL(lib_path)
    
    # Qwen3 model functions
    lib.createQwen3Model.restype = POINTER(Qwen3ModelCStruct)
    lib.createQwen3Model.argtypes = [
        POINTER(Qwen3MetaCStruct),
        POINTER(Qwen3WeightsCStruct),
        DeviceType,
        c_int,
        POINTER(c_int),
    ]
    lib.destroyQwen3Model.argtypes = [POINTER(Qwen3ModelCStruct)]
    lib.createQwen3KVCache.argtypes = [POINTER(Qwen3ModelCStruct)]
    lib.createQwen3KVCache.restype = POINTER(KVCacheCStruct)
    lib.dropQwen3KVCache.argtypes = [POINTER(Qwen3ModelCStruct), POINTER(KVCacheCStruct)]
    lib.inferQwen3Batch.restype = None
    lib.inferQwen3Batch.argtypes = [
        POINTER(Qwen3ModelCStruct),    # struct Qwen3Model const *
        POINTER(c_uint),               # unsigned int const *tokens
        c_uint,                        # unsigned int ntok
        POINTER(c_uint),               # unsigned int const *req_lens
        c_uint,                        # unsigned int nreq
        POINTER(c_uint),               # unsigned int const *req_pos
        POINTER(POINTER(KVCacheCStruct)),  # struct KVCache **kv_caches
        POINTER(c_float),              # float const *temperature (数组!)
        POINTER(c_uint),               # unsigned int const *topk (数组!)
        POINTER(c_float),              # float const *topp (数组!)
        POINTER(c_uint),               # unsigned int *output
    ]
    return lib

LIB = __open_library__()

create_qwen3_model = LIB.createQwen3Model
destroy_qwen3_model = LIB.destroyQwen3Model
create_qwen3_kv_cache = LIB.createQwen3KVCache
drop_qwen3_kv_cache = LIB.dropQwen3KVCache
infer_qwen3_batch = LIB.inferQwen3Batch