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


class InfiniLib:
    def __init__(self, librt, libop):
        self.librt = librt
        self.libop = libop

    def __getattr__(self, name):
        if hasattr(self.libop, name):
            return getattr(self.libop, name)
        elif hasattr(self.librt, name):
            return getattr(self.librt, name)
        else:
            raise AttributeError(f"Attribute {name} not found in library")


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
        libop_path = find_library_in_ld_path("bin", "infiniop.dll")
        librt_path = find_library_in_ld_path("bin", "infinirt.dll")
    elif system_name == "Linux":
        libop_path = find_library_in_ld_path("lib", "libinfiniop.so")
        librt_path = find_library_in_ld_path("lib", "libinfinirt.so")

    assert (
        libop_path is not None
    ), f"Cannot find infiniop.dll or libinfiniop.so. Check if INFINI_ROOT is set correctly."
    assert (
        librt_path is not None
    ), f"Cannot find infinirt.dll or libinfinirt.so. Check if INFINI_ROOT is set correctly."

    librt = ctypes.CDLL(librt_path)
    libop = ctypes.CDLL(libop_path)
    lib = InfiniLib(librt, libop)
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
    lib.infiniopCreateHandle.argtypes = [POINTER(infiniopHandle_t)]
    lib.infiniopCreateHandle.restype = c_int
    lib.infiniopDestroyHandle.argtypes = [infiniopHandle_t]
    lib.infiniopDestroyHandle.restype = c_int
    lib.infinirtSetDevice.argtypes = [c_int, c_int]
    lib.infinirtSetDevice.restype = c_int

    return lib
