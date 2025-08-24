import torch
import ctypes
from ctypes import c_uint64
from libinfiniop import (
    LIBINFINIOP,
    TestTensor,
    get_test_devices,
    check_error,
    test_operator,
    get_args,
    debug_all,
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
# These are not meant to be imported from other modules
_TEST_CASES = [
    # voc, random_val, topp, topk, temperature
    (512, 0.8, 0.8, 3, 0.5),
    (4096, 0.05, 0.9, 5, 1.0),
    (16384, 0.15, 0.85, 10, 2.0),
    (512, 0.08, 0, 3, 0.5),
    (4096, 0.5, 0.9, 1, 1.0),
    (16384, 0.15, 0, 1, 2.0),
    (16384, 0.15, 0, 1, 2.0),
    (32000, 0.08, 0.8, 50, 1.0),
    (32000, 0.08, 1.0, 25, 1.0),
    # (119696, 0.01, 1.0, 100, 1.0),
]

# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.BF16]

_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 0, "rtol": 0},
    InfiniDtype.BF16: {"atol": 0, "rtol": 0},
}


DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def random_sample(data, random_val, topp, topk, voc, temperature):
    if topp > 0 and topk > 1:
        sorted_vals, sorted_indices = torch.sort(data, descending=True)

        scaled_vals = (sorted_vals - sorted_vals[0]) / temperature
        probs = torch.softmax(scaled_vals, dim=0)
        cum_probs = torch.cumsum(probs, dim=0)

        k_index = min(topk, voc) - 1
        threshold = min(cum_probs[k_index], topp) * random_val

        try:
            idx = torch.searchsorted(cum_probs, threshold)
        except Exception:
            # Fallback for manual search if torch.searchsorted is not supported
            indices = (cum_probs >= threshold).nonzero(as_tuple=True)[0]
            idx = (
                indices[0]
                if indices.numel() > 0
                else torch.tensor(len(cum_probs) - 1, device=cum_probs.device)
            )
        return sorted_indices[idx]

    return torch.argmax(data)


def test(
    handle,
    device,
    voc,
    random_val,
    topp,
    topk,
    temperature,
    dtype=InfiniDtype.F16,
    sync=None,
):
    print(
        f"Testing RandomSample on {InfiniDeviceNames[device]} with voc:{voc} random_val:{random_val} topp:{topp} topk:{topk} temperature:{temperature} dtype:{InfiniDtypeNames[dtype]}"
    )

    _perm = torch.randperm(voc)
    logits = TestTensor.from_torch(
        torch.arange(voc)[_perm].float() * 0.0001, dtype, device
    )

    ans = random_sample(
        logits.torch_tensor(), random_val, topp, topk, voc, temperature
    ).to(
        torch.int32
    )  # 这个函数在device速度可能会很慢，可以通过data.to("cpu")方式加快计算过程

    indices = TestTensor([], None, InfiniDtype.I32, device, mode="zeros")

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateRandomSampleDescriptor(
            handle,
            ctypes.byref(descriptor),
            indices.descriptor,
            logits.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [logits, indices]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetRandomSampleWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    def lib_random_sample():
        check_error(
            LIBINFINIOP.infiniopRandomSample(
                descriptor,
                workspace.data(),
                workspace_size.value,
                indices.data(),
                logits.data(),
                random_val,
                topp,
                topk,
                temperature,
                None,
            )
        )

    lib_random_sample()

    if sync is not None:
        sync()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug_all(
            (indices.actual_tensor(), logits.actual_tensor()[indices.actual_tensor()]),
            (ans, logits.torch_tensor()[ans]),
            "or",
            atol=atol,
            rtol=rtol,
        )
    assert (
        indices.actual_tensor() == ans
        or logits.actual_tensor()[indices.actual_tensor()] == logits.torch_tensor()[ans]
    )

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: random_sample(
            logits.torch_tensor(), random_val, topp, topk, voc, temperature
        ), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_random_sample(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyRandomSampleDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    # Execute tests
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
