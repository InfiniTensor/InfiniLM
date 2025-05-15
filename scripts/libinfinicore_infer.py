import ctypes
from ctypes import c_uint, c_int, c_float, c_void_p, POINTER
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


class JiugeMeta(ctypes.Structure):
    _fields_ = [
        ("dt_logits", DataType),
        ("dt_norm", DataType),
        ("dt_mat", DataType),
        ("nlayer", c_uint),
        ("d", c_uint),
        ("nh", c_uint),
        ("nkvh", c_uint),
        ("dh", c_uint),
        ("di", c_uint),
        ("dctx", c_uint),
        ("dvoc", c_uint),
        ("epsilon", c_float),
        ("theta", c_float),
        ("end_token", c_uint),
    ]


# Define the JiugeWeights struct
class JiugeWeights(ctypes.Structure):
    _fields_ = [
        ("nlayer", c_uint),
        ("input_embd", c_void_p),
        ("output_norm", c_void_p),
        ("output_embd", c_void_p),
        ("attn_norm", POINTER(c_void_p)),
        ("attn_qkv", POINTER(c_void_p)),
        ("attn_qkv_b", POINTER(c_void_p)),
        ("attn_o", POINTER(c_void_p)),
        ("ffn_norm", POINTER(c_void_p)),
        ("ffn_gate_up", POINTER(c_void_p)),
        ("ffn_down", POINTER(c_void_p)),
    ]


class JiugeModel(ctypes.Structure):
    pass


class KVCache(ctypes.Structure):
    pass


def __open_library__():
    lib_path = os.path.join(
        os.environ.get("INFINI_ROOT"), "lib", "libinfinicore_infer.so"
    )
    lib = ctypes.CDLL(lib_path)
    lib.createJiugeModel.restype = POINTER(JiugeModel)
    lib.createJiugeModel.argtypes = [
        POINTER(JiugeMeta),  # JiugeMeta const *
        POINTER(JiugeWeights),  # JiugeWeights const *
        DeviceType,  # DeviceType
        c_int,  # int ndev
        POINTER(c_int),  # int const *dev_ids
    ]

    lib.createKVCache.restype = POINTER(KVCache)
    lib.dropKVCache.argtypes = [ctypes.POINTER(JiugeModel), POINTER(KVCache)]
    lib.inferBatch.restype = None
    lib.inferBatch.argtypes = [
        ctypes.POINTER(JiugeModel),  # struct JiugeModel const *
        POINTER(c_uint),  # unsigned int const *tokens
        c_uint,  # unsigned int ntok
        POINTER(c_uint),  # unsigned int const *req_lens
        c_uint,  # unsigned int nreq
        POINTER(c_uint),  # unsigned int const *req_pos
        POINTER(POINTER(KVCache)),  # struct KVCache **kv_caches
        POINTER(c_uint),  # unsigned int *output
        c_float,  # float temperature
        c_uint,  # unsigned int topk
        c_float,  # float topp
    ]

    return lib


LIB = __open_library__()

create_jiuge_model = LIB.createJiugeModel
create_kv_cache = LIB.createKVCache
drop_kv_cache = LIB.dropKVCache
infer_batch = LIB.inferBatch
