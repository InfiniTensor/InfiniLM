import ctypes
from ctypes import c_char, c_char_p, c_size_t, c_uint, c_int, c_float, c_void_p, POINTER
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
    DEVICE_TYPE_KUNLUN = 7
    DEVICE_TYPE_HYGON = 8


class KVCacheCStruct(ctypes.Structure):
    pass


# Model registration system
_model_registry = []


def register_model(model_class):
    """Decorator to register a model class"""
    _model_registry.append(model_class)
    return model_class


def register_lib_functions(lib):
    """Register all model functions with the library"""
    for model_class in _model_registry:
        model_class.register_lib(lib)


class BaseModel:
    def __init__(self):
        self.lib = self._load_library()
        register_lib_functions(self.lib)

    def _load_library(self):
        lib_path = os.path.join(
            os.environ.get("INFINI_ROOT"), "lib", "libinfinicore_infer.so"
        )
        return ctypes.CDLL(lib_path)
