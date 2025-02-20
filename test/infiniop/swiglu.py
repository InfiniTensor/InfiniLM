import torch
import ctypes
from ctypes import POINTER, Structure, c_int32, c_size_t, c_uint64, c_void_p, c_float
from libinfiniop import (
    infiniopHandle_t,
    infiniopTensorDescriptor_t,
    open_lib,
    to_tensor,
    get_test_devices,
    check_error,
    rearrange_if_needed,
    create_workspace,
    test_operator,
    get_args,
    debug,
    get_tolerance,
    profile_operation,
)

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES = [
    # shape, a_stride, b_stride, c_stride
    ((13, 4), None, None, None),
    ((13, 4), (10, 1), (10, 1), (10, 1)),
    ((13, 4, 4), None, None, None),
    ((13, 4, 4), (20, 4, 1), (20, 4, 1), (20, 4, 1)),
    ((16, 5632), None, None, None),
    ((16, 5632), (13312, 1), (13312, 1), (13312, 1)),
    ((4, 4, 5632), None, None, None),
    ((4, 4, 5632), (45056, 5632, 1), (45056, 5632, 1), (45056, 5632, 1)),
]
# Data types used for testing
_TENSOR_DTYPES = [torch.float16, torch.float32]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    torch.float16: {'atol': 0, 'rtol': 1e-2},
    torch.float32: {'atol': 0, 'rtol': 1e-3},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000

class SwiGLUDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopSwiGLUDescriptor_t = POINTER(SwiGLUDescriptor)


def swiglu(a, b):

    return a * b / (1 + torch.exp(-b.float()).to(b.dtype))


def test_out_of_place(
    lib,
    handle,
    torch_device,
    shape,
    a_stride=None,
    b_stride=None,
    c_stride=None,
    dtype=torch.float16,
    sync=None,
):
    print(
        f"Testing SwiGLU on {torch_device} with shape:{shape} a_stride:{a_stride} b_stride:{b_stride} c_stride:{c_stride} dtype:{dtype}"
    )
    a = torch.rand(shape, dtype=dtype).to(torch_device)
    b = torch.rand(shape, dtype=dtype).to(torch_device)
    c = torch.rand(shape, dtype=dtype).to(torch_device)

    ans = swiglu(a, b)

    a, b, c = [
        rearrange_if_needed(tensor, stride)
        for tensor, stride in zip([a, b, c], [a_stride, b_stride, c_stride])
    ]
    a_tensor, b_tensor, c_tensor = [to_tensor(tensor, lib) for tensor in [a, b, c]]


    if sync is not None:
        sync()

    descriptor = infiniopSwiGLUDescriptor_t()
    check_error(
        lib.infiniopCreateSwiGLUDescriptor(
            handle,
            ctypes.byref(descriptor),
            c_tensor.descriptor,
            a_tensor.descriptor,
            b_tensor.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [a_tensor, b_tensor, c_tensor]:
        tensor.descriptor.contents.invalidate()

    def lib_swiglu():
        check_error(
            lib.infiniopSwiGLU(
                descriptor, 
                c_tensor.data, 
                a_tensor.data, 
                b_tensor.data, 
                None
            )
        )
    lib_swiglu()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(c, ans, atol=atol, rtol=rtol)
    assert torch.allclose(c, ans, atol=atol, rtol=rtol)
    print("out-of-place Test passed!")

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: swiglu(a, b), torch_device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_swiglu(), torch_device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(lib.infiniopDestroySwiGLUDescriptor(descriptor))


def test_in_place1(
    lib,
    handle,
    torch_device,
    shape,
    a_stride=None,
    b_stride=None,
    dtype=torch.float16,
    sync=None,
):
    a = torch.rand(shape, dtype=dtype).to(torch_device)
    b = torch.rand(shape, dtype=dtype).to(torch_device)

    ans = swiglu(a, b)

    if sync is not None:
        sync()

    a, b = [
        rearrange_if_needed(tensor, stride)
        for tensor, stride in zip([a, b], [a_stride, b_stride])
    ]
    a_tensor, b_tensor = [to_tensor(tensor, lib) for tensor in [a, b]]

    descriptor = infiniopSwiGLUDescriptor_t()
    
    check_error(
        lib.infiniopCreateSwiGLUDescriptor(
            handle,
            ctypes.byref(descriptor),
            a_tensor.descriptor,
            a_tensor.descriptor,
            b_tensor.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [a_tensor, b_tensor]:
        tensor.descriptor.contents.invalidate()
    def lib_swiglu():
        check_error(
            lib.infiniopSwiGLU(
                descriptor, a_tensor.data, a_tensor.data, b_tensor.data, None
            )
        )
    lib_swiglu()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(a, ans, atol=atol, rtol=rtol)
    assert torch.allclose(a, ans, atol=atol, rtol=rtol)
    print("in-place1 Test passed!")
    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: swiglu(a, b), torch_device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_swiglu(), torch_device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(lib.infiniopDestroySwiGLUDescriptor(descriptor))


def test_in_place2(
    lib,
    handle,
    torch_device,
    shape,
    a_stride=None,
    b_stride=None,
    dtype=torch.float16,
    sync=None,
):
    a = torch.rand(shape, dtype=dtype).to(torch_device)
    b = torch.rand(shape, dtype=dtype).to(torch_device)

    ans = swiglu(a, b)

    if sync is not None:
        sync()

    a, b = [
        rearrange_if_needed(tensor, stride)
        for tensor, stride in zip([a, b], [a_stride, b_stride])
    ]
    a_tensor, b_tensor = [to_tensor(tensor, lib) for tensor in [a, b]]

    descriptor = infiniopSwiGLUDescriptor_t()
    check_error(
        lib.infiniopCreateSwiGLUDescriptor(
            handle,
            ctypes.byref(descriptor),
            b_tensor.descriptor,
            a_tensor.descriptor,
            b_tensor.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [a_tensor, b_tensor]:
        tensor.descriptor.contents.invalidate()

    def lib_swiglu():
        check_error(
            lib.infiniopSwiGLU(
                descriptor, b_tensor.data, a_tensor.data, b_tensor.data, None
            )
        )
    lib_swiglu()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(b, ans, atol=atol, rtol=rtol)
    assert torch.allclose(b, ans, atol=atol, rtol=rtol)
    print("in-place2 Test passed!")
    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: swiglu(a, b), torch_device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_swiglu(), torch_device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(lib.infiniopDestroySwiGLUDescriptor(descriptor))


def test(lib, handle, torch_device, shape, a_stride, b_stride, c_stride, dtype, sync = None):
    test_out_of_place(
        lib, handle, torch_device, shape, a_stride, b_stride, c_stride, dtype, sync
    )
    test_in_place1(lib, handle, torch_device, shape, a_stride, b_stride, dtype, sync)
    test_in_place2(lib, handle, torch_device, shape, a_stride, b_stride, dtype, sync)



if __name__ == "__main__":
    
    args = get_args()
    lib = open_lib()

    lib.infiniopCreateSwiGLUDescriptor.restype = c_int32
    lib.infiniopCreateSwiGLUDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopSwiGLUDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopSwiGLU.restype = c_int32
    lib.infiniopSwiGLU.argtypes = [
        infiniopSwiGLUDescriptor_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroySwiGLUDescriptor.restype = c_int32
    lib.infiniopDestroySwiGLUDescriptor.argtypes = [
        infiniopSwiGLUDescriptor_t,
    ]
    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations
    
    for device in get_test_devices(args):
        test_operator(lib, device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
