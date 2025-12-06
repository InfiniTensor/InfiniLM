#!/usr/bin/env python3
"""
Test script to validate forward pass across different backends and dtypes.

Tests:
1. Python backend with bfloat16
3. C++ backend with bfloat16

This script runs a prefill step (full sequence forward pass with KV cache)
followed by a decode step (single token forward pass with KV cache) and
compares the logits outputs to identify dtype/backend-specific issues.
"""

import infinilm
from infinilm.modeling_utils import get_model_state_dict
from infinilm.cache_utils import DynamicCache
from transformers import AutoTokenizer
import infinicore
import sys
import os
import argparse
import numpy as np
import torch

# Import to_numpy extension for infinicore tensors
try:
    from infinilm.generation.utils import infini_to_numpy
    # This should already be registered, but ensure it's available
    if not hasattr(infinicore.Tensor, 'to_numpy'):
        infinicore.Tensor.to_numpy = infini_to_numpy
except ImportError:
    # If not available, we'll use fallback methods
    pass

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../python"))
test_dir = os.path.dirname(__file__)
sys.path.insert(0, test_dir)


# Import utility functions from test directory
try:
    from utils import infinicore_to_torch_tensor, torch_to_infinicore_tensor
except ImportError:
    # Fallback if utils not available - try to import from parent directory
    try:
        sys.path.insert(0, os.path.join(test_dir, ".."))
        from utils import infinicore_to_torch_tensor, torch_to_infinicore_tensor
    except ImportError:
        print("Warning: Could not import utils. Some conversions may fail.")

        def infinicore_to_torch_tensor(infini_tensor, torch_tensor_for_shape=None):
            """Fallback conversion."""
            return torch.zeros(list(infini_tensor.shape), dtype=torch.float32)

        def torch_to_infinicore_tensor(torch_tensor, infini_device):
            """Fallback conversion."""
            return infinicore.from_list(torch_tensor.tolist())


def get_args():
    parser = argparse.ArgumentParser(
        description="Validate forward pass across backends/dtypes")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device type (default: cuda)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="How are you",
        help="Test prompt (default: 'How are you')",
    )
    parser.add_argument(
        "--num_decode_steps",
        type=int,
        default=2,
        help="Number of decode steps to run after prefill (default: 2)",
    )
    return parser.parse_args()


def create_inputs(prompt, tokenizer, device, backend="cpp"):
    """Create input tensors for forward pass."""
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    # Match examples/llama.py: use encode() without return_tensors to get a list
    input_ids_list = tokenizer.encode(input_content)

    # Create position_ids: [0, 1, 2, ..., seq_len-1]
    seq_len = len(input_ids_list)
    position_ids_list = list(range(seq_len))

    # For Python backend, embedding requires CPU inputs
    # For C++ backend, we can use the specified device
    if backend == "python":
        infini_device = infinicore.device("cpu", 0)
    else:
        infini_device = infinicore.device(device, 0)

    # Match examples/llama.py: use from_list to create tensors
    # Wrap in list to create batch dimension: [[1, 2, 3, ...]]
    input_ids_infini = infinicore.from_list(
        [input_ids_list], device=infini_device)
    # Match generation code: use int64 dtype for position_ids
    position_ids_infini = infinicore.from_list(
        [position_ids_list], dtype=infinicore.int64, device=infini_device)

    return input_ids_infini, position_ids_infini, input_content


def run_forward_pass(model, input_ids, position_ids, backend, dtype, num_decode_steps=2):
    """Run prefill and multiple decode steps with KV cache, return all decode step logits."""
    print(f"  Running forward pass (prefill + {num_decode_steps} decode step(s))...")

    try:
        # Get the underlying model
        if hasattr(model, "_model"):
            underlying_model = model._model
        else:
            underlying_model = model

        # C++ backend has different forward signature - it doesn't accept past_key_values/use_cache
        if backend == "cpp":
            # C++ backend manages its own cache internally
            # Step 1: Prefill - run forward pass with full input sequence
            print(f"    Step 1: Prefill (seq_len={input_ids.shape[1]})...")
            prefill_logits = underlying_model.forward(input_ids, position_ids)

            # Debug: Check tensor before conversion for C++ backend with bfloat16
            if dtype == "bfloat16":
                # Wrap to check properties
                if not hasattr(prefill_logits, "_underlying"):
                    prefill_logits_wrapped = infinicore.Tensor(prefill_logits)
                else:
                    prefill_logits_wrapped = prefill_logits
                print(f"      DEBUG: Prefill logits tensor dtype={prefill_logits_wrapped.dtype}, "
                      f"device={prefill_logits_wrapped.device}, "
                      f"shape={prefill_logits_wrapped.shape}")

            prefill_logits_np = infinicore_to_numpy(prefill_logits)
            print(
                f"    ✓ Prefill completed, logits shape: {prefill_logits_np.shape}")

            # Check prefill logits for issues
            if np.isnan(prefill_logits_np).any():
                print(f"    ⚠ WARNING: Prefill logits contain NaN values!")
                print(f"      NaN count: {np.isnan(prefill_logits_np).sum()}")
                print(
                    f"      Prefill logits stats: min={np.nanmin(prefill_logits_np):.6f}, max={np.nanmax(prefill_logits_np):.6f}, mean={np.nanmean(prefill_logits_np):.6f}")
            if np.isinf(prefill_logits_np).any():
                print(f"    ⚠ WARNING: Prefill logits contain Inf values!")
                print(f"      Inf count: {np.isinf(prefill_logits_np).sum()}")
            if not np.isnan(prefill_logits_np).any():
                print(
                    f"    Prefill logits stats: min={prefill_logits_np.min():.6f}, max={prefill_logits_np.max():.6f}, mean={prefill_logits_np.mean():.6f}")

            # Get device from input_ids
            if hasattr(input_ids, "device"):
                input_device = input_ids.device
            else:
                input_device = getattr(
                    position_ids, "device", infinicore.device("cpu", 0))

            # Initialize decode logits list
            decode_logits_list = []
            seq_len = input_ids.shape[1]
            current_token_id = None

            # Run multiple decode steps
            for decode_step in range(num_decode_steps):
                # Get the predicted token from previous step
                if decode_step == 0:
                    # First decode step: use token from prefill
                    if np.isnan(prefill_logits_np).any():
                        print(f"    ⚠ WARNING: Using default token 29902 due to NaN in prefill logits")
                        current_token_id = 29902
                    else:
                        current_token_id = int(prefill_logits_np.argmax(axis=-1)[0, 0])
                else:
                    # Subsequent decode steps: use token from previous decode
                    prev_logits_np = decode_logits_list[-1]
                    if np.isnan(prev_logits_np).any():
                        print(f"    ⚠ WARNING: Using default token 29902 due to NaN in decode step {decode_step} logits")
                        current_token_id = 29902
                    else:
                        current_token_id = int(prev_logits_np.argmax(axis=-1)[0, 0])

                print(f"    Step {decode_step + 2}: Decode step {decode_step + 1} (next_token_id={current_token_id})...")

                # Create single token input for decode step
                decode_input_ids = infinicore.from_list(
                    [[current_token_id]], device=input_device)

                # Create position_ids for decode step
                decode_position_ids = infinicore.from_list(
                    [[seq_len + decode_step]], dtype=infinicore.int64, device=input_device
                )

                # Run decode step - C++ backend manages cache internally
                decode_logits = underlying_model.forward(
                    decode_input_ids, decode_position_ids)

                # Convert decode logits to numpy
                decode_logits_np = infinicore_to_numpy(decode_logits)
                decode_logits_list.append(decode_logits_np)
                print(f"    ✓ Decode step {decode_step + 1} completed, logits shape: {decode_logits_np.shape}")

                # Check decode logits for issues
                if np.isnan(decode_logits_np).any():
                    print(f"    ⚠ WARNING: Decode step {decode_step + 1} logits contain NaN values!")
                    print(f"      NaN count: {np.isnan(decode_logits_np).sum()}")
                if np.isinf(decode_logits_np).any():
                    print(f"    ⚠ WARNING: Decode step {decode_step + 1} logits contain Inf values!")
                    print(f"      Inf count: {np.isinf(decode_logits_np).sum()}")
                if not np.isnan(decode_logits_np).any():
                    print(f"    Decode step {decode_step + 1} logits stats: min={decode_logits_np.min():.6f}, max={decode_logits_np.max():.6f}, mean={decode_logits_np.mean():.6f}")
        else:
            # Python backend uses DynamicCache
            # Get model config
            if hasattr(model, "config"):
                model_config = model.config
            elif hasattr(underlying_model, "config"):
                model_config = underlying_model.config
            else:
                raise ValueError("Model does not have a config attribute")

            # Create KV cache
            past_key_values = DynamicCache(config=model_config)

            # Step 1: Prefill - run forward pass with full input sequence
            print(f"    Step 1: Prefill (seq_len={input_ids.shape[1]})...")
            prefill_logits = underlying_model.forward(
                input_ids, position_ids, past_key_values=past_key_values, use_cache=True
            )
            prefill_logits_np = infinicore_to_numpy(prefill_logits)
            print(
                f"    ✓ Prefill completed, logits shape: {prefill_logits_np.shape}")

            # Get device from input_ids
            if hasattr(input_ids, "device"):
                input_device = input_ids.device
            else:
                # Fallback: try to get device from position_ids or use CPU
                input_device = getattr(
                    position_ids, "device", infinicore.device("cpu", 0))

            # Initialize decode logits list
            decode_logits_list = []
            seq_len = input_ids.shape[1]
            current_token_id = None

            # Run multiple decode steps
            for decode_step in range(num_decode_steps):
                # Get the predicted token from previous step
                if decode_step == 0:
                    # First decode step: use token from prefill
                    if np.isnan(prefill_logits_np).any():
                        print(f"    ⚠ WARNING: Using default token 29902 due to NaN in prefill logits")
                        current_token_id = 29902
                    else:
                        current_token_id = int(prefill_logits_np.argmax(axis=-1)[0, 0])
                else:
                    # Subsequent decode steps: use token from previous decode
                    prev_logits_np = decode_logits_list[-1]
                    if np.isnan(prev_logits_np).any():
                        print(f"    ⚠ WARNING: Using default token 29902 due to NaN in decode step {decode_step} logits")
                        current_token_id = 29902
                    else:
                        current_token_id = int(prev_logits_np.argmax(axis=-1)[0, 0])

                print(f"    Step {decode_step + 2}: Decode step {decode_step + 1} (next_token_id={current_token_id})...")

                # Create single token input for decode step
                decode_input_ids = infinicore.from_list(
                    [[current_token_id]], device=input_device)

                # Create position_ids for decode step
                decode_position_ids = infinicore.from_list(
                    [[seq_len + decode_step]], dtype=infinicore.int64, device=input_device
                )

                # Run decode step with KV cache
                decode_logits = underlying_model.forward(
                    decode_input_ids, decode_position_ids, past_key_values=past_key_values, use_cache=True
                )

                # Convert decode logits to numpy
                decode_logits_np = infinicore_to_numpy(decode_logits)
                decode_logits_list.append(decode_logits_np)
                print(f"    ✓ Decode step {decode_step + 1} completed, logits shape: {decode_logits_np.shape}")

                # Check decode logits for issues
                if np.isnan(decode_logits_np).any():
                    print(f"    ⚠ WARNING: Decode step {decode_step + 1} logits contain NaN values!")
                    print(f"      NaN count: {np.isnan(decode_logits_np).sum()}")
                if np.isinf(decode_logits_np).any():
                    print(f"    ⚠ WARNING: Decode step {decode_step + 1} logits contain Inf values!")
                    print(f"      Inf count: {np.isinf(decode_logits_np).sum()}")
                if not np.isnan(decode_logits_np).any():
                    print(f"    Decode step {decode_step + 1} logits stats: min={decode_logits_np.min():.6f}, max={decode_logits_np.max():.6f}, mean={decode_logits_np.mean():.6f}")

        # Summary of all decode steps
        print(f"  ✓ Forward pass completed (prefill + {num_decode_steps} decode step(s))")
        for i, logits_np in enumerate(decode_logits_list):
            print(f"    Decode step {i + 1} logits shape: {logits_np.shape}, dtype: {logits_np.dtype}")

        # Check for issues in all decode steps
        has_error = False
        for i, logits_np in enumerate(decode_logits_list):
            if np.isnan(logits_np).any():
                print(f"    ⚠ WARNING: Decode step {i + 1} logits contain NaN values!")
                print(f"      NaN count: {np.isnan(logits_np).sum()}")
                has_error = True
            if np.isinf(logits_np).any():
                print(f"    ⚠ WARNING: Decode step {i + 1} logits contain Inf values!")
                print(f"      Inf count: {np.isinf(logits_np).sum()}")
                has_error = True
            if np.abs(logits_np).max() < 1.0:
                print(f"    ⚠ WARNING: Decode step {i + 1} logits are very small (max abs: {np.abs(logits_np).max():.6f})")

        # Get predicted token from last decode step
        if decode_logits_list and not np.isnan(decode_logits_list[-1]).any():
            predicted_token = int(decode_logits_list[-1].argmax(axis=-1)[0, 0])
            print(f"    Predicted token ID from decode step {num_decode_steps}: {predicted_token}")

        # Return tuple of all decode logits
        return tuple(decode_logits_list), has_error

    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return None, True


def infinicore_to_numpy(tensor):
    """Convert infinicore tensor to numpy array."""
    # Wrap raw C++ tensor in Python Tensor wrapper if needed
    # C++ backend returns raw _infinicore.Tensor, Python backend returns infinicore.Tensor
    if not hasattr(tensor, "_underlying"):
        # It's a raw C++ tensor, wrap it in the Python Tensor class
        tensor = infinicore.Tensor(tensor)

    # Move tensor to CPU if it's on a device (required for conversion)
    if tensor.device.type != "cpu":
        tensor_cpu = tensor.to(infinicore.device("cpu", 0))
    else:
        tensor_cpu = tensor

    # Handle bfloat16 specially - convert to float32 via torch first
    # (to_numpy doesn't support bfloat16 directly)
    if tensor_cpu.dtype == infinicore.bfloat16:
        import ctypes
        # Ensure tensor is actually on CPU and contiguous
        if tensor_cpu.device.type != "cpu":
            print(
                f"      DEBUG: WARNING - tensor_cpu.device.type={tensor_cpu.device.type}, forcing CPU move")
            tensor_cpu = tensor_cpu.to(infinicore.device("cpu", 0))
        if not tensor_cpu.is_contiguous():
            tensor_cpu = tensor_cpu.contiguous()

        # Read raw data as uint16 (bfloat16 storage format)
        # IMPORTANT: Ensure we're reading from CPU memory
        data_ptr = tensor_cpu.data_ptr()
        num_elements = tensor_cpu.numel()
        shape = tensor_cpu.shape

        # Debug: Check data pointer and device
        print(
            f"      DEBUG: Reading bfloat16 data: data_ptr={data_ptr}, num_elements={num_elements}, shape={shape}, device={tensor_cpu.device}")

        # Use a safer approach: copy data using ctypes.memmove to ensure we read from CPU memory
        uint16_array = np.empty(shape, dtype=np.uint16)
        ctypes.memmove(uint16_array.ctypes.data, data_ptr,
                       num_elements * 2)  # 2 bytes per uint16

        # Convert to torch bfloat16, then to float32, then to numpy
        torch_uint16 = torch.from_numpy(uint16_array)
        torch_bf16 = torch_uint16.view(torch.bfloat16)
        torch_f32 = torch_bf16.float()
        result = torch_f32.numpy()

        # Debug: Check for NaN in conversion result
        if np.isnan(result).any():
            print(f"      DEBUG: NaN detected after bfloat16->float32 conversion")
            print(f"        NaN count: {np.isnan(result).sum()}/{result.size}")
            print(
                f"        uint16_array stats: min={uint16_array.min()}, max={uint16_array.max()}, mean={uint16_array.mean():.2f}")
            print(
                f"        torch_bf16 stats: min={torch_bf16.min().item():.6f}, max={torch_bf16.max().item():.6f}, mean={torch_bf16.mean().item():.6f}")
            print(
                f"        torch_f32 stats: min={torch_f32.min().item():.6f}, max={torch_f32.max().item():.6f}, mean={torch_f32.mean().item():.6f}")

        return result

    # For other dtypes, use the to_numpy method
    result = tensor_cpu.to_numpy()

    # Debug: Check for NaN in conversion result
    if np.isnan(result).any():
        print(
            f"      DEBUG: NaN detected after to_numpy conversion (dtype={tensor_cpu.dtype})")
        print(f"        NaN count: {np.isnan(result).sum()}/{result.size}")

    return result


def test_configuration(model_path, device, backend, dtype, prompt, num_decode_steps=2):
    """Test a specific backend/dtype configuration."""
    print("\n" + "=" * 80)
    print(f"Testing: Backend={backend}, Dtype={dtype}")
    print("=" * 80)

    # Parse dtype
    if dtype == "bfloat16":
        infini_dtype = infinicore.bfloat16
    elif dtype == "float32":
        infini_dtype = infinicore.float32
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # For Python backend, always use CPU (embedding layer requires CPU inputs)
    # For C++ backend, use the specified device
    if backend == "python":
        infini_device = infinicore.device("cpu", 0)
    else:
        infini_device = infinicore.device(device, 0)

    # Load tokenizer
    print("\n1. Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"  ✓ Tokenizer loaded")
    except Exception as e:
        print(f"  ✗ Failed to load tokenizer: {e}")
        return None, True

    # Create model
    print(f"\n2. Creating model (backend={backend}, dtype={dtype})...")
    try:
        model = infinilm.AutoLlamaModel.from_pretrained(
            model_path, device=infini_device, dtype=infini_dtype, backend=backend
        )
        print(f"  ✓ Model created")
    except Exception as e:
        print(f"  ✗ Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return None, True

    # Load weights
    print(f"\n3. Loading model weights...")
    try:
        model_param_infini = get_model_state_dict(
            model_path,
            device=infini_device,
            dtype=infini_dtype,
        )
        model.load_state_dict(model_param_infini)
        print(f"  ✓ Weights loaded")
    except Exception as e:
        print(f"  ✗ Failed to load weights: {e}")
        import traceback
        traceback.print_exc()
        return None, True

    # Create inputs
    print(f"\n4. Creating inputs from prompt: '{prompt}'...")
    try:
        input_ids, position_ids, input_content = create_inputs(
            prompt, tokenizer, device, backend=backend)
        print(f"  ✓ Inputs created")
        print(f"    Input content: {input_content[:100]}...")
        print(f"    Input shape: {input_ids.shape}")
        print(
            f"    Input device: {input_ids.device.type if hasattr(input_ids, 'device') else 'unknown'}")
    except Exception as e:
        print(f"  ✗ Failed to create inputs: {e}")
        import traceback
        traceback.print_exc()
        return None, True

    # Run forward pass (prefill + multiple decode steps)
    print(f"\n5. Running forward pass (prefill + {num_decode_steps} decode step(s))...")
    logits_tuple, has_error = run_forward_pass(
        model, input_ids, position_ids, backend, dtype, num_decode_steps)

    if has_error:
        return None, True

    return logits_tuple, False


def compare_logits(logits1, logits2, name1, name2, step_name="logits"):
    """Compare two logits arrays."""
    print(f"\n{'=' * 80}")
    print(f"Comparing: {name1} vs {name2} ({step_name})")
    print(f"{'=' * 80}")

    if logits1 is None or logits2 is None:
        print(f"  ✗ Cannot compare: one or both {step_name} are None")
        return False

    if logits1.shape != logits2.shape:
        print(f"  ✗ Shape mismatch: {logits1.shape} vs {logits2.shape}")
        return False

    # Compute differences
    diff = np.abs(logits1 - logits2)
    max_diff = diff.max()
    mean_diff = diff.mean()

    print(f"  Max absolute difference: {max_diff:.6f}")
    print(f"  Mean absolute difference: {mean_diff:.6f}")

    # Check if they're close (allowing for dtype differences)
    # For bfloat16 vs float32, we expect larger differences
    rtol = 1e-2  # 1% relative tolerance
    atol = 1.0   # Absolute tolerance

    is_close = np.allclose(logits1, logits2, rtol=rtol, atol=atol)

    if is_close:
        print(f"  ✓ {step_name.capitalize()} are close (within tolerance)")
    else:
        print(f"  ⚠ {step_name.capitalize()} differ significantly")
        # Show top differences
        flat_diff = diff.flatten()
        top_indices = np.argsort(flat_diff)[-10:][::-1]
        print(f"  Top 10 differences:")
        for idx in top_indices:
            pos = np.unravel_index(idx, diff.shape)
            print(
                f"    Position {pos}: {logits1[pos]:.6f} vs {logits2[pos]:.6f}, diff={diff[pos]:.6f}")

    return is_close


def main():
    args = get_args()

    print("=" * 80)
    print("Forward Pass Validation Test")
    print("=" * 80)
    print(f"Model path: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"Prompt: {args.prompt}")
    print(f"Number of decode steps: {args.num_decode_steps}")
    print("=" * 80)

    results = {}

    # Test 1: Python backend with bfloat16
    print("\n\n" + "=" * 80)
    print("TEST 1: Python Backend + BFloat16")
    print("=" * 80)
    logits_py_bf16, error = test_configuration(
        args.model_path, args.device, "python", "bfloat16", args.prompt, args.num_decode_steps
    )
    results["python_bf16"] = (logits_py_bf16, error)

    # Test 3: C++ backend with bfloat16
    print("\n\n" + "=" * 80)
    print("TEST 3: C++ Backend + BFloat16")
    print("=" * 80)
    logits_cpp_bf16, error = test_configuration(
        args.model_path, args.device, "cpp", "bfloat16", args.prompt, args.num_decode_steps
    )
    results["cpp_bf16"] = (logits_cpp_bf16, error)

    # Compare results
    print("\n\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    comparisons = []

    # Compare Python BF16 vs C++ BF16 (should be similar)
    if not results["python_bf16"][1] and not results["cpp_bf16"][1]:
        py_logits = results["python_bf16"][0]
        cpp_logits = results["cpp_bf16"][0]

        if py_logits is not None and cpp_logits is not None:
            # Compare all decode steps
            num_steps = min(len(py_logits), len(cpp_logits))
            for step_idx in range(num_steps):
                step_name = f"decode step {step_idx + 1}"
                is_close = compare_logits(
                    py_logits[step_idx],
                    cpp_logits[step_idx],
                    "Python BF16",
                    "C++ BF16",
                    step_name
                )
                comparisons.append((f"Python BF16 vs C++ BF16 ({step_name})", is_close))

    # Summary
    print("\n\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for name, (logits, error) in results.items():
        status = "✗ ERROR" if error else "✓ SUCCESS"
        print(f"{name:20s}: {status}")

    print("\nComparisons:")
    for name, is_close in comparisons:
        status = "✓ CLOSE" if is_close else "⚠ DIFFERENT"
        print(f"  {name:30s}: {status}")

    # Final verdict
    all_success = all(not error for _, (_, error) in results.items())
    if all_success:
        print("\n✓ All tests completed successfully")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
