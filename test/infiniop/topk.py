import torch
import ctypes
from ctypes import c_uint64, c_int32
from libinfiniop import (
    LIBINFINIOP,
    TestTensor,
    get_test_devices,
    check_error,
    test_operator,
    get_args,
    debug,
    get_tolerance,
    profile_operation,
    TestWorkspace,
    InfiniDtype,
    InfiniDtypeNames,
    InfiniDeviceNames,
    infiniopOperatorDescriptor_t,
)

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
_TEST_CASES = [
    # x_shape, k, axis, largest, sorted
    ((1, 10), 3, -1, True, True),
    ((16, 2048), 10, 1, True, True),
    ((16, 2048), 10, 1, False, True),
    ((16, 2048), 1, 1, True, True),  # k=1 is an edge case
    ((4, 8, 128), 5, 2, True, True),
    ((4, 8, 128), 128, 2, True, True),  # k equals dimension size
    ((4, 8, 128), 5, 1, True, True),
    ((4, 8, 128), 5, 0, True, True),
]

# x types used for testing
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.BF16, InfiniDtype.F32]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-2, "rtol": 1e-2},
    InfiniDtype.BF16: {"atol": 1e-1, "rtol": 1e-1},
    InfiniDtype.F32: {"atol": 1e-6, "rtol": 1e-6},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def test(
    handle,
    device,
    x_shape,
    k,
    axis,
    largest,
    sorted,
    dtype=InfiniDtype.F16,
    sync=None,
):
    print(
        f"Testing TopK on {InfiniDeviceNames[device]} with x_shape:{x_shape} k:{k} axis:{axis} largest:{largest} sorted:{sorted} dtype:{InfiniDtypeNames[dtype]}"
    )

    # Calculate output shape
    output_shape = list(x_shape)
    _axis = axis
    if _axis < 0:
        _axis += len(x_shape)
    output_shape[_axis] = k
    output_shape = tuple(output_shape)

    x = TestTensor(x_shape, None, dtype, device)
    values = TestTensor(output_shape, None, dtype, device, mode="zeros")
    indices = TestTensor(output_shape, None, InfiniDtype.I64, device, mode="zeros")

    torch_values, torch_indices = torch.topk(
        x.torch_tensor(), k, dim=axis, largest=largest, sorted=sorted
    )

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()

    check_error(
        LIBINFINIOP.infiniopCreateTopKDescriptor(
            handle,
            ctypes.byref(descriptor),
            values.descriptor,
            indices.descriptor,
            x.descriptor,
            k,
            axis,
            largest,
            sorted,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [x, values, indices]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetTopKWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, x.device)

    def lib_topk():
        check_error(
            LIBINFINIOP.infiniopTopK(
                descriptor,
                workspace.data(),
                workspace_size.value,
                values.data(),
                indices.data(),
                x.data(),
                None,
            )
        )

    lib_topk()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        print("Values:")
        debug(values.actual_tensor(), torch_values, atol=atol, rtol=rtol)
        print("Indices:")
        debug(indices.actual_tensor(), torch_indices, atol=0, rtol=0)

    assert torch.allclose(values.actual_tensor(), torch_values, atol=atol, rtol=rtol)
    assert torch.equal(indices.actual_tensor(), torch_indices)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: torch.topk(x.torch_tensor(), k, dim=axis, largest=largest, sorted=sorted), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_topk(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyTopKDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    # Execute tests
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m") 