"""
Utility functions for InfiniLM Llama model tests.

This module provides shared utility functions for tensor conversion,
parameter name normalization, and tensor comparison.
"""

from typing import Tuple, Dict, Callable, Optional, Any, List
import torch

try:
    import infinicore
except ImportError:
    infinicore = None


def normalize_param_name(name: str) -> str:
    """Normalize parameter name (remove 'model.' prefix if present)"""
    if name.startswith("model."):
        return name[6:]  # Remove "model." prefix
    return name


def to_infinicore_dtype(torch_dtype):
    """Convert PyTorch data type to infinicore data type"""
    if infinicore is None:
        raise ImportError("InfiniCore package not found")

    if torch_dtype == torch.float32:
        return infinicore.float32
    elif torch_dtype == torch.float16:
        return infinicore.float16
    elif torch_dtype == torch.bfloat16:
        return infinicore.bfloat16
    elif torch_dtype == torch.int8:
        return infinicore.int8
    elif torch_dtype == torch.int16:
        return infinicore.int16
    elif torch_dtype == torch.int32:
        return infinicore.int32
    elif torch_dtype == torch.int64:
        return infinicore.int64
    elif torch_dtype == torch.uint8:
        return infinicore.uint8
    elif torch_dtype == torch.bool:
        return infinicore.bool
    else:
        raise ValueError(f"Unsupported torch dtype: {torch_dtype}")


def torch_to_infinicore_tensor(torch_tensor, infini_device):
    """
    Convert PyTorch tensor to InfiniCore tensor.

    Args:
        torch_tensor: PyTorch tensor
        infini_device: InfiniCore device object

    Returns:
        InfiniCore tensor
    """
    if infinicore is None:
        raise ImportError("InfiniCore package not found")

    # Ensure tensor is contiguous (but keep it on its current device)
    torch_tensor = torch_tensor.contiguous()

    # Convert dtype
    infini_dtype = to_infinicore_dtype(torch_tensor.dtype)

    # Create InfiniCore tensor from torch tensor's data pointer
    if torch_tensor.is_contiguous():
        return infinicore.from_blob(
            torch_tensor.data_ptr(),
            list(torch_tensor.shape),
            dtype=infini_dtype,
            device=infini_device,
        )
    else:
        return infinicore.strided_from_blob(
            torch_tensor.data_ptr(),
            list(torch_tensor.shape),
            list(torch_tensor.stride()),
            dtype=infini_dtype,
            device=infini_device,
        )


def to_torch_dtype(infini_dtype):
    """Convert InfiniCore data type to PyTorch data type"""
    if infinicore is None:
        raise ImportError("InfiniCore package not found")

    # infini_dtype is a dtype object from infinicore.dtype
    # Access the underlying enum value for comparison
    from infinicore.lib import _infinicore

    # Get underlying enum value
    if hasattr(infini_dtype, '_underlying'):
        underlying = infini_dtype._underlying
    else:
        # If it's not a dtype object, try to use it directly
        underlying = infini_dtype

    # Compare underlying enum values
    if underlying == _infinicore.DataType.F32:
        return torch.float32
    elif underlying == _infinicore.DataType.F16:
        return torch.float16
    elif underlying == _infinicore.DataType.BF16:
        return torch.bfloat16
    elif underlying == _infinicore.DataType.I8:
        return torch.int8
    elif underlying == _infinicore.DataType.I16:
        return torch.int16
    elif underlying == _infinicore.DataType.I32:
        return torch.int32
    elif underlying == _infinicore.DataType.I64:
        return torch.int64
    elif underlying == _infinicore.DataType.U8:
        return torch.uint8
    elif underlying == _infinicore.DataType.BOOL:
        return torch.bool
    else:
        raise ValueError(
            f"Unsupported infinicore dtype: {infini_dtype} (underlying enum: {underlying})")


def infinicore_to_torch_tensor(infini_tensor, torch_reference):
    """
    Convert InfiniCore tensor to PyTorch tensor for comparison.

    Args:
        infini_tensor: InfiniCore tensor (can be raw C++ tensor or Python wrapper)
        torch_reference: PyTorch tensor reference (for shape and device)

    Returns:
        PyTorch tensor with InfiniCore data on the same device as torch_reference
    """
    if infinicore is None:
        raise ImportError("InfiniCore package not found")

    # Wrap raw C++ tensor in Python Tensor wrapper if needed
    # get_parameter returns a raw _infinicore.Tensor, but we need infinicore.Tensor
    if not hasattr(infini_tensor, '_underlying'):
        # It's a raw C++ tensor, wrap it in the Python Tensor class
        infini_tensor = infinicore.Tensor(infini_tensor)

    # Get device from reference tensor
    ref_device = torch_reference.device

    # Determine target InfiniCore device
    if ref_device.type == "cuda":
        target_infini_device = infinicore.device("cuda", ref_device.index)
    else:
        target_infini_device = infinicore.device("cpu", 0)

    # Ensure source tensor is on the target device and contiguous
    # This is important when GPU support is compiled - we need to explicitly
    # move tensors to the correct device and make them contiguous
    # When GPU support is compiled but we're using CPU, we need to be extra careful
    try:
        # For CPU, always ensure tensor is explicitly on CPU and contiguous
        if ref_device.type == "cpu":
            cpu_device = infinicore.device("cpu", 0)
            # Move to CPU if not already there
            if hasattr(infini_tensor, 'device'):
                source_device = infini_tensor.device
                if str(source_device) != str(cpu_device):
                    infini_tensor = infini_tensor.to(cpu_device)
            # Ensure contiguous
            if not infini_tensor.is_contiguous():
                infini_tensor = infini_tensor.contiguous()
        else:
            # For GPU, ensure on target device and contiguous
            if hasattr(infini_tensor, 'device'):
                source_device = infini_tensor.device
                source_device_str = str(source_device)
                target_device_str = str(target_infini_device)
                if source_device_str != target_device_str:
                    infini_tensor = infini_tensor.to(target_infini_device)
            if not infini_tensor.is_contiguous():
                infini_tensor = infini_tensor.contiguous()
    except Exception as e:
        # If device operations fail, try to ensure contiguous at least
        if hasattr(infini_tensor, 'is_contiguous') and not infini_tensor.is_contiguous():
            infini_tensor = infini_tensor.contiguous()

    # Create a PyTorch tensor with the same shape, dtype, and device as reference
    torch_result = torch.zeros(
        list(infini_tensor.shape),
        dtype=to_torch_dtype(infini_tensor.dtype),
        device=ref_device,
    )

    # For CPU, use a workaround: create an intermediate tensor and copy through it
    # This avoids issues with rearrange when GPU support is compiled
    if ref_device.type == "cpu":
        # Check if source tensor is on CUDA - if so, we need pinned memory
        source_is_cuda = False
        source_cuda_device = None
        if hasattr(infini_tensor, 'device'):
            source_device = infini_tensor.device
            source_device_str = str(source_device)
            source_is_cuda = source_device_str.startswith("cuda")
            if source_is_cuda:
                # Extract CUDA device index from device string (e.g., "cuda:0")
                try:
                    cuda_index = int(source_device_str.split(
                        ":")[1]) if ":" in source_device_str else 0
                    source_cuda_device = infinicore.device("cuda", cuda_index)
                except:
                    source_cuda_device = infinicore.device("cuda", 0)

        # If source is on CUDA, we need to ensure the intermediate CPU tensor
        # uses pinned memory. The copy_from function will handle setting the
        # CUDA context, but we need to create the intermediate with pin_memory=True
        # so it gets pinned host memory that CUDA can safely copy to.
        # Note: The empty() function will check the current runtime when pin_memory=True.
        # Since copy_from sets the context to CUDA before copying, we create the
        # intermediate with pin_memory=True, and even if it initially gets regular
        # memory, the copy operation should still work. However, for better performance
        # and reliability, we try to use .to() method which handles device transfers more safely.

        # Try using .to() method first, which handles device transfers internally
        try:
            # Use .to() to move tensor to CPU - this should handle the transfer safely
            cpu_tensor = infini_tensor.to(target_infini_device)
            if not cpu_tensor.is_contiguous():
                cpu_tensor = cpu_tensor.contiguous()

            # Create temp tensor from PyTorch and copy from the CPU tensor
            temp_tensor = torch_to_infinicore_tensor(
                torch_result, target_infini_device)
            temp_tensor.copy_(cpu_tensor)
        except Exception as e:
            # Fallback: create intermediate tensor and copy through it
            # Create an intermediate contiguous tensor on CPU
            # Use pin_memory=True if source is CUDA to ensure proper D2H copy
            intermediate = infinicore.empty(
                list(infini_tensor.shape),
                dtype=infini_tensor.dtype,
                device=target_infini_device,
                pin_memory=source_is_cuda  # Pin memory if copying from CUDA
            )

            # Copy source to intermediate first
            try:
                intermediate.copy_(infini_tensor)
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to copy tensor to intermediate: {e2}")

            # Now create temp tensor from PyTorch and copy from intermediate
            temp_tensor = torch_to_infinicore_tensor(
                torch_result, target_infini_device)
            temp_tensor.copy_(intermediate)
    else:
        # For GPU, use direct copy
        temp_tensor = torch_to_infinicore_tensor(
            torch_result, target_infini_device)
        temp_tensor.copy_(infini_tensor)

    return torch_result


def tensor_all_close(tensor1: torch.Tensor, tensor2: torch.Tensor,
                     rtol: float = 1e-5, atol: float = 1e-5) -> Tuple[bool, Dict]:
    """
    Compare two tensors for approximate equality.

    Args:
        tensor1: First tensor to compare
        tensor2: Second tensor to compare
        rtol: Relative tolerance (default: 1e-5)
        atol: Absolute tolerance (default: 1e-5)

    Returns:
        Tuple of (is_close, stats_dict) where stats_dict contains:
        - max_abs_diff: Maximum absolute difference
        - mean_abs_diff: Mean absolute difference
        - max_rel_diff: Maximum relative difference
        - is_close: Boolean indicating if tensors are close
        - has_nan: Boolean indicating if either tensor has NaN
        - has_inf: Boolean indicating if either tensor has Inf
    """
    if tensor1.shape != tensor2.shape:
        return False, {"error": "Shape mismatch", "shape1": tensor1.shape, "shape2": tensor2.shape}

    # Check for NaN/Inf values
    tensor1_has_nan = torch.isnan(tensor1).any().item()
    tensor1_has_inf = torch.isinf(tensor1).any().item()
    tensor2_has_nan = torch.isnan(tensor2).any().item()
    tensor2_has_inf = torch.isinf(tensor2).any().item()

    has_nan = tensor1_has_nan or tensor2_has_nan
    has_inf = tensor1_has_inf or tensor2_has_inf

    # If either tensor has NaN/Inf, handle specially
    if has_nan or has_inf:
        # Compute stats only on finite values
        finite_mask = torch.isfinite(tensor1) & torch.isfinite(tensor2)

        if finite_mask.any():
            diff = (tensor1 - tensor2).abs()
            finite_diff = diff[finite_mask]
            max_diff = finite_diff.max().item() if len(finite_diff) > 0 else float('nan')
            mean_diff = finite_diff.mean().item() if len(finite_diff) > 0 else float('nan')

            # For relative diff, use finite values from tensor2
            finite_tensor2 = tensor2[finite_mask]
            if len(finite_tensor2) > 0:
                relative_max_diff = (
                    finite_diff / finite_tensor2.abs().clamp(min=1e-8)).max().item()
            else:
                relative_max_diff = float('nan')
        else:
            max_diff = float('nan')
            mean_diff = float('nan')
            relative_max_diff = float('nan')

        is_close = False  # Can't be close if there are NaN/Inf
    else:
        # Normal comparison when no NaN/Inf
        diff = (tensor1 - tensor2).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        relative_max_diff = (diff / tensor2.abs().clamp(min=1e-8)).max().item()
        is_close = torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)

    stats = {
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "max_rel_diff": relative_max_diff,
        "is_close": is_close,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "tensor1_has_nan": tensor1_has_nan,
        "tensor1_has_inf": tensor1_has_inf,
        "tensor2_has_nan": tensor2_has_nan,
        "tensor2_has_inf": tensor2_has_inf,
    }

    return is_close, stats


def validate_infinicore_component(
    op_name: str,
    infinicore_op: Callable,
    transformers_input: torch.Tensor,
    transformers_output: torch.Tensor,
    infinicore_input: torch.Tensor,
    infinicore_output: torch.Tensor,
    infini_device: Any,
    op_kwargs: Optional[Dict[str, Any]] = None,
    tolerance: float = 1e-5,
    debug_callback: Optional[Callable] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Validate an InfiniCore component by comparing it with Transformers implementation.

    This function implements the pattern from section 9d2b:
    1. Test 1: Run InfiniCore ops with InfiniCore input (current behavior)
    2. Test 2: Run InfiniCore ops with Transformers input (eliminate input diff)
    3. Compare Test 2 output with Transformers output to verify ops implementation
    4. Compare Test 1 vs Test 2 to see impact of input difference

    Args:
        op_name: Name of the operation (for logging)
        infinicore_op: InfiniCore operation function (e.g., F.rms_norm)
        transformers_input: Input tensor from Transformers model
        transformers_output: Output tensor from Transformers model
        infinicore_input: Input tensor from InfiniLM model
        infinicore_output: Output tensor from InfiniLM model
        infini_device: InfiniCore device object
        op_kwargs: Additional keyword arguments to pass to the InfiniCore op
        tolerance: Tolerance for comparison (default: 1e-5)
        debug_callback: Optional callback function for detailed debugging
                       Signature: debug_callback(trans_input, infini_input, trans_output,
                                                  infini_output, test1_output, test2_output)
        verbose: Whether to print detailed output (default: True)

    Returns:
        Dictionary containing validation results:
        - test1_match: Whether Test 1 output matches InfiniLM output
        - test2_match: Whether Test 2 output matches Transformers output
        - ops_correct: Whether InfiniCore ops implementation is correct (Test 2 result)
        - input_impact: Impact of input difference (Test 1 vs Test 2)
        - test1_stats: Statistics for Test 1 comparison
        - test2_stats: Statistics for Test 2 comparison
        - input_diff_stats: Statistics for input difference analysis
    """
    if op_kwargs is None:
        op_kwargs = {}

    results = {
        "test1_match": False,
        "test2_match": False,
        "ops_correct": False,
        "input_impact": "unknown",
        "test1_stats": {},
        "test2_stats": {},
        "input_diff_stats": {},
    }

    try:
        if verbose:
            print(
                f"\n   Validating {op_name} with InfiniCore ops using real data...")

        # Convert inputs to InfiniCore tensors
        infini_input_tensor = torch_to_infinicore_tensor(
            infinicore_input, infini_device)
        trans_input_tensor = torch_to_infinicore_tensor(
            transformers_input, infini_device)

        # Test 1: Call InfiniCore ops with InfiniCore input (current behavior)
        if verbose:
            print(f"\n   Test 1: InfiniCore ops with InfiniCore input...")

        # Prepare arguments for the op
        # For ops that take multiple inputs, we need to handle them
        # This assumes the op takes input as first arg and kwargs
        test1_inputs = [infini_input_tensor]
        test1_output = infinicore_op(*test1_inputs, **op_kwargs)
        test1_output_torch = infinicore_to_torch_tensor(
            test1_output, infinicore_output)

        # Compare Test 1 with InfiniLM output
        test1_match, test1_stats = tensor_all_close(
            test1_output_torch, infinicore_output, rtol=tolerance, atol=tolerance
        )
        results["test1_match"] = test1_match
        results["test1_stats"] = test1_stats

        if verbose:
            if test1_match:
                print(f"      ✓ Test 1: InfiniCore ops matches InfiniLM output")
            else:
                print(f"      ⚠ Test 1: InfiniCore ops differs from InfiniLM output")
                print(
                    f"         Max abs diff: {test1_stats['max_abs_diff']:.15f}")
                print(
                    f"         Mean abs diff: {test1_stats['mean_abs_diff']:.15f}")

        # Test 2: Call InfiniCore ops with Transformers input (to eliminate input diff)
        if verbose:
            print(
                f"\n   Test 2: InfiniCore ops with Transformers input (eliminating input diff)...")

        test2_inputs = [trans_input_tensor]
        test2_output = infinicore_op(*test2_inputs, **op_kwargs)
        test2_output_torch = infinicore_to_torch_tensor(
            test2_output, transformers_output)

        # Compare Test 2 (InfiniCore ops with Transformers input) vs Transformers output
        if verbose:
            print(
                f"\n   Test 2 Results: InfiniCore ops (Transformers input) vs Transformers output:")

        test2_match, test2_stats = tensor_all_close(
            test2_output_torch, transformers_output, rtol=tolerance, atol=tolerance
        )
        results["test2_match"] = test2_match
        results["test2_stats"] = test2_stats
        results["ops_correct"] = test2_match

        if verbose:
            print(f"      Max abs diff: {test2_stats['max_abs_diff']:.15f}")
            print(f"      Mean abs diff: {test2_stats['mean_abs_diff']:.15f}")
            print(f"      Max rel diff: {test2_stats['max_rel_diff']:.15f}")

            if test2_match:
                print(
                    f"      ✓ InfiniCore ops matches Transformers when using same input!")
            else:
                print(
                    f"      ⚠ InfiniCore ops still differs from Transformers even with same input")
                print(
                    f"         This suggests the {op_name} computation itself differs")

                # Find max diff position
                diff = (test2_output_torch - transformers_output).abs()
                max_diff_idx = diff.argmax()
                max_diff_pos = torch.unravel_index(max_diff_idx, diff.shape)
                if verbose:
                    print(f"\n      Max diff position {max_diff_pos}:")
                    print(
                        f"         Transformers: {transformers_output[max_diff_pos].item():.15f}")
                    print(
                        f"         InfiniCore ops (Trans input): {test2_output_torch[max_diff_pos].item():.15f}")
                    print(
                        f"         Difference: {diff[max_diff_pos].item():.15f}")

        # Compare Test 1 vs Test 2 to see impact of input difference
        if verbose:
            print(f"\n   Comparing Test 1 vs Test 2 (impact of input difference):")

        test1_vs_test2_diff = (test1_output_torch - test2_output_torch).abs()
        test1_vs_test2_max = test1_vs_test2_diff.max().item()
        test1_vs_test2_mean = test1_vs_test2_diff.mean().item()

        results["input_diff_stats"] = {
            "max_abs_diff": test1_vs_test2_max,
            "mean_abs_diff": test1_vs_test2_mean,
        }

        if verbose:
            print(f"      Max abs diff: {test1_vs_test2_max:.15f}")
            print(f"      Mean abs diff: {test1_vs_test2_mean:.15f}")

        if test1_vs_test2_max > tolerance:
            results["input_impact"] = "significant"
            if verbose:
                print(f"      ⚠ Input difference causes significant output difference")
        else:
            results["input_impact"] = "minimal"
            if verbose:
                print(f"      ✓ Input difference has minimal impact on output")

        # Compare input data between Transformers and InfiniCore
        if verbose:
            print(f"\n   Comparing input data (Transformers vs InfiniCore):")

        input_diff = (transformers_input - infinicore_input).abs()
        input_diff_max = input_diff.max().item()
        input_diff_mean = input_diff.mean().item()

        results["input_diff_stats"]["input_max_diff"] = input_diff_max
        results["input_diff_stats"]["input_mean_diff"] = input_diff_mean

        if verbose:
            print(f"   Input diff stats: min={input_diff.min().item():.15f}, "
                  f"max={input_diff_max:.15f}, mean={input_diff_mean:.15f}")

            if input_diff_max > 1e-6:
                max_input_diff_idx = input_diff.argmax()
                max_input_diff_pos = torch.unravel_index(
                    max_input_diff_idx, input_diff.shape)
                print(f"   ⚠ Max input diff at position {max_input_diff_pos}:")
                print(
                    f"      Transformers: {transformers_input[max_input_diff_pos].item():.15f}")
                print(
                    f"      InfiniCore: {infinicore_input[max_input_diff_pos].item():.15f}")
                print(
                    f"      Difference: {input_diff[max_input_diff_pos].item():.15f}")
            else:
                print(f"   ✓ Input data matches (within tolerance)")

        # Call debug callback if provided
        if debug_callback is not None:
            try:
                debug_callback(
                    transformers_input, infinicore_input,
                    transformers_output, infinicore_output,
                    test1_output_torch, test2_output_torch
                )
            except Exception as e:
                if verbose:
                    print(f"   ⚠ Debug callback failed: {e}")

        # Summary
        if verbose:
            print(f"\n   Summary:")
            print(
                f"      Test 1 (InfiniCore input): {'✓ PASS' if test1_match else '✗ FAIL'}")
            print(
                f"      Test 2 (Transformers input): {'✓ PASS' if test2_match else '✗ FAIL'}")
            print(
                f"      InfiniCore ops correctness: {'✓ CORRECT' if results['ops_correct'] else '✗ INCORRECT'}")
            print(f"      Input impact: {results['input_impact']}")

    except Exception as e:
        if verbose:
            print(f"   ✗ Validation failed with exception: {e}")
            import traceback
            traceback.print_exc()
        results["error"] = str(e)

    return results
