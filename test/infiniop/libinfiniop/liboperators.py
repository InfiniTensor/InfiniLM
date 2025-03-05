import os
import platform
import ctypes
from ctypes import c_int, c_int64, c_uint64, Structure, POINTER
from .datatypes import *
from .devices import *
from pathlib import Path

Device = c_int
Optype = c_int

INFINI_ROOT = os.getenv("INFINI_ROOT") or str(Path.home() / ".infini")


class TensorDescriptor(Structure):
    _fields_ = []


infiniopTensorDescriptor_t = ctypes.POINTER(TensorDescriptor)


class CTensor:
    def __init__(self, desc, torch_tensor):
        self.descriptor = desc
        self.torch_tensor_ = torch_tensor
        self.data = torch_tensor.data_ptr()
    
    def destroyDesc(self, lib_):
        lib_.infiniopDestroyTensorDescriptor(self.descriptor)
        self.descriptor = None


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
    lib = ctypes.CDLL(library_path)
    lib.infiniopCreateTensorDescriptor.argtypes = [
        POINTER(infiniopTensorDescriptor_t),
        c_uint64,
        POINTER(c_uint64),
        POINTER(c_int64),
        c_int,
    ]
    lib.infiniopCreateTensorDescriptor.restype = c_int
    lib.infiniopDestroyTensorDescriptor.argtypes = [infiniopTensorDescriptor_t]
    lib.infiniopDestroyTensorDescriptor.restype = c_int
    lib.infiniopCreateHandle.argtypes = [POINTER(infiniopHandle_t), c_int, c_int]
    lib.infiniopCreateHandle.restype = c_int
    lib.infiniopDestroyHandle.argtypes = [infiniopHandle_t]
    lib.infiniopDestroyHandle.restype = c_int

    return lib
