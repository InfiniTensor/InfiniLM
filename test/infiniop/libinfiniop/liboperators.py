from calendar import c
import os
import platform
import ctypes
from ctypes import c_int, c_int64, c_uint64, Structure, POINTER, c_size_t
from .datatypes import *
from .devices import *

Device = c_int
Optype = c_int

INFINI_ROOT = os.environ.get("INFINI_ROOT")


class TensorDescriptor(Structure):
    _fields_ = [
        ("dtype", c_int),
        ("ndim", c_size_t),
        ("shape", POINTER(c_size_t)),
        ("strides", POINTER(c_int64)),
    ]

    def invalidate(self):
        for i in range(self.ndim):
            self.shape[i] = 0
            self.strides[i] = 0


infiniopTensorDescriptor_t = ctypes.POINTER(TensorDescriptor)


class CTensor:
    def __init__(self, desc, data):
        self.descriptor = desc
        self.data = data


class Handle(Structure):
    _fields_ = [("device", c_int), ("device_id", c_int)]


infiniopHandle_t = POINTER(Handle)


# Open operators library
def open_lib():
    def find_library_in_ld_path(subdir, library_name):
        ld_library_path = os.path.join(INFINI_ROOT, subdir)
        paths = ld_library_path.split(os.pathsep)
        for path in paths:
            full_path = os.path.join(path, library_name)
            if os.path.isfile(full_path):
                return full_path
        return None

    system_name = platform.system()
    # Load the library
    if system_name == "Windows":
        library_path = find_library_in_ld_path("bin", "infiniop.dll")
    elif system_name == "Linux":
        library_path = find_library_in_ld_path("lib", "libinfiniop.so")

    assert (
        library_path is not None
    ), f"Cannot find infiniop.dll or libinfiniop.so. Check if INFINI_ROOT is set correctly."
    ctypes.CDLL(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin\cudnn64_9.dll")
    lib = ctypes.CDLL(library_path)
    lib.infiniopCreateTensorDescriptor.argtypes = [
        POINTER(infiniopTensorDescriptor_t),
        c_uint64,
        POINTER(c_uint64),
        POINTER(c_int64),
        c_int,
    ]
    lib.infiniopCreateHandle.argtypes = [POINTER(infiniopHandle_t), c_int, c_int]
    lib.infiniopCreateHandle.restype = c_int
    lib.infiniopDestroyHandle.argtypes = [infiniopHandle_t]
    lib.infiniopDestroyHandle.restype = c_int

    return lib
