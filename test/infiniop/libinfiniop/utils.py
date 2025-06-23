import ctypes
from .datatypes import *
from .devices import *
from typing import Sequence
from .liboperators import infiniopTensorDescriptor_t, CTensor, infiniopHandle_t


def check_error(status):
    if status != 0:
        raise Exception("Error code " + str(status))


def to_tensor(tensor, lib, force_unsigned=False):
    """
    Convert a PyTorch tensor to a library Tensor(descriptor, data).
    """
    import torch

    ndim = tensor.ndimension()
    shape = (ctypes.c_size_t * ndim)(*tensor.shape)
    strides = (ctypes.c_int64 * ndim)(*(tensor.stride()))
    # fmt: off
    dt = (
        InfiniDtype.I8 if tensor.dtype == torch.int8 else
        InfiniDtype.I16 if tensor.dtype == torch.int16 else
        InfiniDtype.I32 if tensor.dtype == torch.int32 else
        InfiniDtype.I64 if tensor.dtype == torch.int64 else
        InfiniDtype.U8 if tensor.dtype == torch.uint8 else
        InfiniDtype.F16 if tensor.dtype == torch.float16 else
        InfiniDtype.BF16 if tensor.dtype == torch.bfloat16 else
        InfiniDtype.F32 if tensor.dtype == torch.float32 else
        InfiniDtype.F64 if tensor.dtype == torch.float64 else
        # TODO: These following types may not be supported by older
        # versions of PyTorch.
        InfiniDtype.U16 if tensor.dtype == torch.uint16 else
        InfiniDtype.U32 if tensor.dtype == torch.uint32 else
        InfiniDtype.U64 if tensor.dtype == torch.uint64 else
        None
    )
    
    if force_unsigned:
        dt = (
            InfiniDtype.U8 if dt == InfiniDtype.I8 else
            InfiniDtype.U16 if dt == InfiniDtype.I16 else
            InfiniDtype.U32 if dt == InfiniDtype.I32 else
            InfiniDtype.U64 if dt == InfiniDtype.I64 else
            dt
        )

    # fmt: on
    assert dt is not None
    # Create TensorDecriptor
    tensor_desc = infiniopTensorDescriptor_t()
    lib.infiniopCreateTensorDescriptor(
        ctypes.byref(tensor_desc), ndim, shape, strides, dt
    )
    # Create Tensor
    return CTensor(tensor_desc, tensor)


def create_workspace(size, torch_device):
    print(f" - Workspace Size : {size}")
    if size == 0:
        return None
    import torch

    return torch.zeros(size=(size,), dtype=torch.uint8, device=torch_device)


def create_handle(lib):
    handle = infiniopHandle_t()
    check_error(lib.infiniopCreateHandle(ctypes.byref(handle)))
    return handle


def destroy_handle(lib, handle):
    check_error(lib.infiniopDestroyHandle(handle))


def rearrange_tensor(tensor, new_strides):
    """
    Given a PyTorch tensor and a list of new strides, return a new PyTorch tensor with the given strides.
    """
    import torch

    shape = tensor.shape

    new_size = [0] * len(shape)
    left = 0
    right = 0
    for i in range(len(shape)):
        if new_strides[i] > 0:
            new_size[i] = (shape[i] - 1) * new_strides[i] + 1
            right += new_strides[i] * (shape[i] - 1)
        else:  # TODO: Support negative strides in the future
            # new_size[i] = (shape[i] - 1) * (-new_strides[i]) + 1
            # left += new_strides[i] * (shape[i] - 1)
            raise ValueError("Negative strides are not supported yet")

    # Create a new tensor with zeros
    new_tensor = torch.zeros(
        (right - left + 1,), dtype=tensor.dtype, device=tensor.device
    )

    # Generate indices for original tensor based on original strides
    indices = [torch.arange(s) for s in shape]
    mesh = torch.meshgrid(*indices, indexing="ij")

    # Flatten indices for linear indexing
    linear_indices = [m.flatten() for m in mesh]

    # Calculate new positions based on new strides
    new_positions = sum(
        linear_indices[i] * new_strides[i] for i in range(len(shape))
    ).to(tensor.device)
    offset = -left
    new_positions += offset

    # Copy the original data to the new tensor
    new_tensor.view(-1).index_add_(0, new_positions, tensor.view(-1))
    new_tensor.set_(new_tensor.untyped_storage(), offset, shape, tuple(new_strides))

    return new_tensor


def rearrange_if_needed(tensor, stride):
    """
    Rearrange a PyTorch tensor if the given stride is not None.
    """
    return rearrange_tensor(tensor, stride) if stride is not None else tensor


def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="Test Operator")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Whether profile tests",
    )
    parser.add_argument(
        "--num_prerun",
        type=lambda x: max(0, int(x)),
        default=10,
        help="Set the number of pre-runs before profiling. Default is 10. Must be a non-negative integer.",
    )
    parser.add_argument(
        "--num_iterations",
        type=lambda x: max(0, int(x)),
        default=1000,
        help="Set the number of iterations for profiling. Default is 1000. Must be a non-negative integer.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to turn on debug mode. If turned on, it will display detailed information about the tensors and discrepancies.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run CPU test",
    )
    parser.add_argument(
        "--nvidia",
        action="store_true",
        help="Run NVIDIA GPU test",
    )
    parser.add_argument(
        "--cambricon",
        action="store_true",
        help="Run Cambricon MLU test",
    )
    parser.add_argument(
        "--ascend",
        action="store_true",
        help="Run ASCEND NPU test",
    )
    parser.add_argument(
        "--metax",
        action="store_true",
        help="Run METAX GPU test",
    )
    parser.add_argument(
        "--moore",
        action="store_true",
        help="Run MTHREADS GPU test",
    )
    parser.add_argument(
        "--kunlun",
        action="store_true",
        help="Run KUNLUN XPU test",
    )

    return parser.parse_args()


def synchronize_device(torch_device):
    import torch

    if torch_device == "cuda":
        torch.cuda.synchronize()
    elif torch_device == "npu":
        torch.npu.synchronize()
    elif torch_device == "mlu":
        torch.mlu.synchronize()


def debug(actual, desired, atol=0, rtol=1e-2, equal_nan=False, verbose=True):
    """
    Debugging function to compare two tensors (actual and desired) and print discrepancies.
    Arguments:
    ----------
    - actual : The tensor containing the actual computed values.
    - desired : The tensor containing the expected values that `actual` should be compared to.
    - atol : optional (default=0)
        The absolute tolerance for the comparison.
    - rtol : optional (default=1e-2)
        The relative tolerance for the comparison.
    - equal_nan : bool, optional (default=False)
        If True, `NaN` values in `actual` and `desired` will be considered equal.
    - verbose : bool, optional (default=True)
        If True, the function will print detailed information about any discrepancies between the tensors.
    """
    import numpy as np

    print_discrepancy(actual, desired, atol, rtol, equal_nan, verbose)
    np.testing.assert_allclose(
        actual.cpu(), desired.cpu(), rtol, atol, equal_nan, verbose=True
    )


def debug_all(
    actual_vals: Sequence,
    desired_vals: Sequence,
    condition: str,
    atol=0,
    rtol=1e-2,
    equal_nan=False,
    verbose=True,
):
    """
    Debugging function to compare two sequences of values (actual and desired) pair by pair, results
    are linked by the given logical condition, and prints discrepancies
    Arguments:
    ----------
    - actual_vals (Sequence): A sequence (e.g., list or tuple) of actual computed values.
    - desired_vals (Sequence): A sequence (e.g., list or tuple) of desired (expected) values to compare against.
    - condition (str): A string specifying the condition for passing the test. It must be either:
        - 'or': Test passes if any pair of actual and desired values satisfies the tolerance criteria.
        - 'and': Test passes if all pairs of actual and desired values satisfy the tolerance criteria.
    - atol (float, optional): Absolute tolerance. Default is 0.
    - rtol (float, optional): Relative tolerance. Default is 1e-2.
    - equal_nan (bool, optional): If True, NaN values in both actual and desired are considered equal. Default is False.
    - verbose (bool, optional): If True, detailed output is printed for each comparison. Default is True.
    Raises:
    ----------
    - AssertionError: If the condition is not satisfied based on the provided `condition`, `atol`, and `rtol`.
    - ValueError: If the length of `actual_vals` and `desired_vals` do not match.
    - AssertionError: If the specified `condition` is not 'or' or 'and'.
    """
    assert len(actual_vals) == len(desired_vals), "Invalid Length"
    assert condition in {
        "or",
        "and",
    }, "Invalid condition: should be either 'or' or 'and'"
    import numpy as np

    passed = False if condition == "or" else True

    for index, (actual, desired) in enumerate(zip(actual_vals, desired_vals)):
        print(f" \033[36mCondition #{index + 1}:\033[0m {actual} == {desired}")
        indices = print_discrepancy(actual, desired, atol, rtol, equal_nan, verbose)
        if condition == "or":
            if not passed and len(indices) == 0:
                passed = True
        elif condition == "and":
            if passed and len(indices) != 0:
                passed = False
                print(
                    f"\033[31mThe condition has not been satisfied: Condition #{index + 1}\033[0m"
                )
            np.testing.assert_allclose(
                actual.cpu(),
                desired.cpu(),
                rtol,
                atol,
                equal_nan,
                verbose=True,
                strict=True,
            )
    assert passed, "\033[31mThe condition has not been satisfied\033[0m"


def print_discrepancy(actual, expected, atol=0, rtol=1e-3, equal_nan=True, verbose=True):
    if actual.shape != expected.shape:
        raise ValueError("Tensors must have the same shape to compare.")

    import torch
    import sys

    is_terminal = sys.stdout.isatty()

    actual_isnan = torch.isnan(actual)
    expected_isnan = torch.isnan(expected)

    # Calculate the difference mask based on atol and rtol
    nan_mismatch = actual_isnan ^ expected_isnan if equal_nan else actual_isnan | expected_isnan
    diff_mask = nan_mismatch | (torch.abs(actual - expected) > (atol + rtol * torch.abs(expected)))
    diff_indices = torch.nonzero(diff_mask, as_tuple=False)
    delta = actual - expected

    # Display format: widths for columns
    col_width = [18, 20, 20, 20]
    decimal_places = [0, 12, 12, 12]
    total_width = sum(col_width) + sum(decimal_places)

    def add_color(text, color_code):
        if is_terminal:
            return f"\033[{color_code}m{text}\033[0m"
        else:
            return text

    if verbose:
        for idx in diff_indices:
            index_tuple = tuple(idx.tolist())
            actual_str = f"{actual[index_tuple]:<{col_width[1]}.{decimal_places[1]}f}"
            expected_str = (
                f"{expected[index_tuple]:<{col_width[2]}.{decimal_places[2]}f}"
            )
            delta_str = f"{delta[index_tuple]:<{col_width[3]}.{decimal_places[3]}f}"
            print(
                f" > Index: {str(index_tuple):<{col_width[0]}}"
                f"actual: {add_color(actual_str, 31)}"
                f"expect: {add_color(expected_str, 32)}"
                f"delta: {add_color(delta_str, 33)}"
            )

        print(add_color(" INFO:", 35))
        print(f"  - Actual dtype: {actual.dtype}")
        print(f"  - Desired dtype: {expected.dtype}")
        print(f"  - Atol: {atol}")
        print(f"  - Rtol: {rtol}")
        print(
            f"  - Mismatched elements: {len(diff_indices)} / {actual.numel()} ({len(diff_indices) / actual.numel() * 100}%)"
        )
        print(
            f"  - Min(actual) : {torch.min(actual):<{col_width[1]}} | Max(actual) : {torch.max(actual):<{col_width[2]}}"
        )
        print(
            f"  - Min(desired): {torch.min(expected):<{col_width[1]}} | Max(desired): {torch.max(expected):<{col_width[2]}}"
        )
        print(
            f"  - Min(delta)  : {torch.min(delta):<{col_width[1]}} | Max(delta)  : {torch.max(delta):<{col_width[2]}}"
        )
        print("-" * total_width + "\n")

    return diff_indices


def get_tolerance(tolerance_map, tensor_dtype, default_atol=0, default_rtol=1e-3):
    """
    Returns the atol and rtol for a given tensor data type in the tolerance_map.
    If the given data type is not found, it returns the provided default tolerance values.
    """
    return tolerance_map.get(
        tensor_dtype, {"atol": default_atol, "rtol": default_rtol}
    ).values()


def timed_op(func, num_iterations, device):
    import time

    """ Function for timing operations with synchronization. """
    synchronize_device(device)
    start = time.time()
    for _ in range(num_iterations):
        func()
    synchronize_device(device)
    return (time.time() - start) / num_iterations


def profile_operation(desc, func, torch_device, NUM_PRERUN, NUM_ITERATIONS):
    """
    Unified profiling workflow that is used to profile the execution time of a given function.
    It first performs a number of warmup runs, then performs timed execution and
    prints the average execution time.

    Arguments:
    ----------
    - desc (str): Description of the operation, used for output display.
    - func (callable): The operation function to be profiled.
    - torch_device (str): The device on which the operation runs, provided for timed execution.
    - NUM_PRERUN (int): The number of warmup runs.
    - NUM_ITERATIONS (int): The number of timed execution iterations, used to calculate the average execution time.
    """
    # Warmup runs
    for _ in range(NUM_PRERUN):
        func()

    # Timed execution
    elapsed = timed_op(lambda: func(), NUM_ITERATIONS, torch_device)
    print(f" {desc} time: {elapsed * 1000 :6f} ms")


def test_operator(lib, device, test_func, test_cases, tensor_dtypes):
    """
    Testing a specified operator on the given device with the given test function, test cases, and tensor data types.

    Arguments:
    ----------
    - lib (ctypes.CDLL): The library object containing the operator implementations.
    - device (InfiniDeviceEnum): The device on which the operator should be tested. See device.py.
    - test_func (function): The test function to be executed for each test case.
    - test_cases (list of tuples): A list of test cases, where each test case is a tuple of parameters
        to be passed to `test_func`.
    - tensor_dtypes (list): A list of tensor data types (e.g., `torch.float32`) to test.
    """
    lib.infinirtSetDevice(device, ctypes.c_int(0))
    handle = create_handle(lib)
    try:
        for test_case in test_cases:
            for tensor_dtype in tensor_dtypes:
                test_func(
                    lib,
                    handle,
                    infiniDeviceEnum_str_map[device],
                    *test_case,
                    tensor_dtype,
                    get_sync_func(device)
                )
    finally:
        destroy_handle(lib, handle)


def get_test_devices(args):
    """
    Using the given parsed Namespace to determine the devices to be tested.

    Argument:
    - args: the parsed Namespace object.

    Return:
    - devices_to_test: the devices that will be tested. Default is CPU.
    """
    devices_to_test = []

    if args.cpu:
        devices_to_test.append(InfiniDeviceEnum.CPU)
    if args.nvidia:
        devices_to_test.append(InfiniDeviceEnum.NVIDIA)
    if args.cambricon:
        import torch_mlu

        devices_to_test.append(InfiniDeviceEnum.CAMBRICON)
    if args.ascend:
        import torch
        import torch_npu

        torch.npu.set_device(0)  # Ascend NPU needs explicit device initialization
        devices_to_test.append(InfiniDeviceEnum.ASCEND)
    if args.metax:
        import torch

        devices_to_test.append(InfiniDeviceEnum.METAX)
    if args.moore:
        import torch
        import torch_musa

        devices_to_test.append(InfiniDeviceEnum.MOORE)
    if args.kunlun:
        import torch_xmlir

        devices_to_test.append(InfiniDeviceEnum.KUNLUN)
    if not devices_to_test:
        devices_to_test = [InfiniDeviceEnum.CPU]

    return devices_to_test


def get_sync_func(device):
    import torch
    device_str = infiniDeviceEnum_str_map[device]
    
    if device == InfiniDeviceEnum.CPU:
        sync = None
    else:
        sync = getattr(torch, device_str).synchronize
    
    return sync
