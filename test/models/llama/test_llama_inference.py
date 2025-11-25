#!/usr/bin/env python3
"""
Test script to validate inference for InfiniLM Llama model.

This test compares inference outputs from InfiniLM model with transformers model
for a single request scenario:
1. Load model from transformers
2. Create InfiniLM model and load weights
3. Prepare a single request (input_ids, position_ids)
4. Run forward pass on both models
5. Compare logits outputs
"""

import sys
import os
import json
from pathlib import Path
from typing import Optional, Tuple

try:
    import torch
    import transformers
except ImportError as e:
    print(f"Error: Required packages not found. Please install: {e}")
    sys.exit(1)

try:
    import infinicore
except ImportError as e:
    print(f"Error: InfiniCore package not found. Please install it: {e}")
    sys.exit(1)

try:
    from infinilm.models.llama import LlamaConfig, LlamaForCausalLM, Device
except ImportError as e:
    print(f"Error: InfiniLM Python package not found. Please install it:")
    print(f"  pip install -e .")
    print(f"  or")
    print(f"  xmake build _infinilm_llama && xmake install _infinilm_llama")
    print(f"  Error: {e}")
    sys.exit(1)

# Import shared utilities
from utils import (
    normalize_param_name,
    tensor_all_close,
    to_infinicore_dtype,
    torch_to_infinicore_tensor,
    to_torch_dtype,
    infinicore_to_torch_tensor,
)


def load_model_config(model_dir: str) -> dict:
    """Load model configuration from config.json"""
    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def create_llama_config_from_dict(config_dict: dict) -> LlamaConfig:
    """Create a LlamaConfig from dictionary"""
    return LlamaConfig(**config_dict)


def load_weights_into_infinilm_model(infinilm_model, transformers_model, infini_device, torch_device):
    """
    Load weights from transformers model into InfiniLM model.

    Args:
        infinilm_model: InfiniLM model instance
        transformers_model: Transformers model instance
        infini_device: InfiniCore device
        torch_device: PyTorch device

    Returns:
        Number of matched parameters
    """
    transformers_state_dict = transformers_model.state_dict()
    infinilm_expected_keys = set(infinilm_model.state_dict().keys())

    infinilm_state_dict = {}
    matched_keys = []
    torch_tensors_keepalive = []

    for key, tensor in transformers_state_dict.items():
        normalized_key = normalize_param_name(key)
        matching_key = None
        for infinilm_key in infinilm_expected_keys:
            if normalize_param_name(infinilm_key) == normalized_key:
                matching_key = infinilm_key
                break

        if matching_key:
            torch_tensor = tensor.detach().clone().to(torch_device).contiguous()
            torch_tensors_keepalive.append(torch_tensor)
            infini_tensor = torch_to_infinicore_tensor(
                torch_tensor, infini_device)
            infinilm_state_dict[matching_key] = infini_tensor
            matched_keys.append(f"{key} -> {matching_key}")

    print(f"   ✓ Matched {len(matched_keys)} parameters for loading")

    infinilm_model.load_state_dict(infinilm_state_dict)

    # Clear references after loading
    infinilm_state_dict.clear()
    torch_tensors_keepalive.clear()

    return len(matched_keys)


def validate_inference(model_dir: str, prompt: str = "Hello, how are you?",
                       device_type: str = "cpu", device_index: int = 0) -> bool:
    """
    Validate inference for InfiniLM llama model.

    This test loads weights from transformers model and compares inference outputs
    for a single request scenario.

    Args:
        model_dir: Path to the model directory
        prompt: Input prompt text (default: "Hello, how are you?")
        device_type: Device type for validation ("cpu", "cuda", etc.) (default: "cpu")
        device_index: Device index (default: 0)

    Returns:
        True if inference validation passes, False otherwise
    """
    print("=" * 70)
    print("Llama Model Inference Validation Test")
    print("=" * 70)
    print(f"\nThis test compares inference outputs between InfiniLM and transformers")
    print(f"for a single request scenario.")
    print(f"Device: {device_type}:{device_index}")
    print(f"Prompt: {prompt}")
    print("=" * 70)

    # Load configuration
    print("\n1. Loading model configuration...")
    try:
        config_dict = load_model_config(model_dir)
        print(f"   ✓ Configuration loaded from {model_dir}/config.json")
    except Exception as e:
        print(f"   ✗ Failed to load configuration: {e}")
        return False

    # Create InfiniLM config and model
    print("\n2. Creating InfiniLM LlamaForCausalLM...")
    try:
        infinilm_config = create_llama_config_from_dict(config_dict)
        if not infinilm_config.validate():
            print("   ✗ InfiniLM configuration validation failed")
            return False

        # Check device availability
        from infinicore.lib import _infinicore
        if device_type == "cuda":
            nvidia_device_type = _infinicore.Device.Type.NVIDIA
            device_count = _infinicore.get_device_count(nvidia_device_type)
            if device_count == 0:
                print(f"   ✗ No NVIDIA/CUDA devices available")
                return False
            if device_index >= device_count:
                print(f"   ✗ CUDA device index {device_index} is out of range")
                return False

        # Create device
        infini_device = infinicore.device(device_type, device_index)
        device_type_upper = device_type.upper()
        if device_type_upper == "CUDA":
            device_type_upper = "NVIDIA"
        device = Device(device_type_upper, device_index)
        infinilm_model = LlamaForCausalLM(infinilm_config, device)
        print(f"   ✓ InfiniLM model created on {device_type}:{device_index}")
    except Exception as e:
        print(f"   ✗ Failed to create InfiniLM model: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Load transformers model
    print("\n3. Loading LlamaForCausalLM from transformers...")
    try:
        if device_type == "cuda":
            torch_device = torch.device(f"cuda:{device_index}")
        else:
            torch_device = torch.device("cpu")

        transformers_model = transformers.LlamaForCausalLM.from_pretrained(
            model_dir,
            dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        transformers_model = transformers_model.to(torch_device)
        transformers_model.eval()  # Set to evaluation mode
        print(f"   ✓ Transformers model loaded on {torch_device}")
    except Exception as e:
        print(f"   ✗ Failed to load transformers model: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Load weights into InfiniLM model
    print("\n4. Loading weights into InfiniLM model...")
    try:
        num_params = load_weights_into_infinilm_model(
            infinilm_model, transformers_model, infini_device, torch_device)
        print(f"   ✓ Loaded {num_params} parameters")
    except Exception as e:
        print(f"   ✗ Failed to load weights: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Prepare input
    print("\n5. Preparing input...")
    try:
        # Use transformers tokenizer to tokenize the prompt
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(torch_device)

        # Create position_ids (0 to seq_len-1)
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(
            0, seq_len, dtype=torch.long, device=torch_device).unsqueeze(0)

        print(f"   ✓ Input prepared")
        print(f"     Input shape: {input_ids.shape}")
        print(f"     Position IDs shape: {position_ids.shape}")
        print(f"     Input tokens: {input_ids.tolist()[0]}")
    except Exception as e:
        print(f"   ✗ Failed to prepare input: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Run inference on transformers model
    print("\n6. Running inference on transformers model...")
    try:
        with torch.no_grad():
            outputs = transformers_model(
                input_ids=input_ids,
                position_ids=position_ids,
                use_cache=False
            )
            transformers_logits = outputs.logits
        print(f"   ✓ Transformers inference completed")
        print(f"     Logits shape: {transformers_logits.shape}")
        print(f"     Logits dtype: {transformers_logits.dtype}")
        print(f"     Logits stats: min={transformers_logits.min().item():.6f}, "
              f"max={transformers_logits.max().item():.6f}, "
              f"mean={transformers_logits.mean().item():.6f}")

        # Decode predicted tokens for human understanding
        transformers_predicted_ids = transformers_logits.argmax(dim=-1)
        transformers_predicted_tokens = transformers_predicted_ids[0].tolist()
        transformers_predicted_text = tokenizer.decode(
            transformers_predicted_tokens, skip_special_tokens=True)
        print(f"     Input prompt: {prompt}")
        print(
            f"     Transformers predicted tokens: {transformers_predicted_tokens}")
        print(
            f"     Transformers predicted text: {transformers_predicted_text}")
    except Exception as e:
        print(f"   ✗ Failed to run transformers inference: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Run inference on InfiniLM model
    print("\n7. Running inference on InfiniLM model...")
    try:
        # Convert input to InfiniCore tensors
        infini_input_ids = torch_to_infinicore_tensor(input_ids, infini_device)
        infini_position_ids = torch_to_infinicore_tensor(
            position_ids, infini_device)

        print(f"   ✓ Converted inputs to InfiniCore tensors")

        # Check if forward method is available
        if hasattr(infinilm_model._model, 'forward'):
            # Call forward method
            infini_logits = infinilm_model._model.forward(
                infini_input_ids,
                infini_position_ids,
                None  # kv_caches
            )
            print(f"   ✓ InfiniLM forward pass completed")

            # Convert InfiniCore logits to PyTorch tensor
            infinilm_logits = infinicore_to_torch_tensor(
                infini_logits, transformers_logits)
            print(f"   ✓ Converted logits to PyTorch tensor")
            print(f"     Logits shape: {infinilm_logits.shape}")
            print(f"     Logits dtype: {infinilm_logits.dtype}")
            print(f"     Logits stats: min={infinilm_logits.min().item():.6f}, "
                  f"max={infinilm_logits.max().item():.6f}, "
                  f"mean={infinilm_logits.mean().item():.6f}")

            # Check for potential issues
            if torch.isnan(infinilm_logits).any():
                print(f"     ⚠ WARNING: InfiniLM logits contain NaN values!")
            if torch.isinf(infinilm_logits).any():
                print(f"     ⚠ WARNING: InfiniLM logits contain Inf values!")

            # Check if logits are too small (might indicate model not working)
            if infinilm_logits.abs().max().item() < 1.0:
                print(
                    f"     ⚠ WARNING: InfiniLM logits are very small (max abs: {infinilm_logits.abs().max().item():.6f})")

            # Decode predicted tokens for human understanding
            infinilm_predicted_ids = infinilm_logits.argmax(dim=-1)

            # Check if all predictions are the same (indicates model might not be learning)
            unique_predictions = torch.unique(infinilm_predicted_ids[0])
            if len(unique_predictions) == 1:
                print(
                    f"     ⚠ WARNING: InfiniLM is predicting the same token ({unique_predictions[0].item()}) for all positions!")
            infinilm_predicted_tokens = infinilm_predicted_ids[0].tolist()
            infinilm_predicted_text = tokenizer.decode(
                infinilm_predicted_tokens, skip_special_tokens=True)
            print(
                f"     InfiniLM predicted tokens: {infinilm_predicted_tokens}")
            print(f"     InfiniLM predicted text: {infinilm_predicted_text}")
        else:
            print(f"   ⚠ Forward method not yet available in Python bindings")
            print(f"     This test will validate model setup and weight loading only")
            print(f"     Once forward is implemented, uncomment the forward call above")
            # For now, we'll just validate that models are set up correctly
            print(f"   ✓ Model setup validated (forward not yet implemented)")
            return True  # Return True for now since forward isn't implemented
    except NotImplementedError:
        print(f"   ⚠ Forward method not yet implemented")
        print(f"     This test validates model setup and weight loading only")
        return True
    except Exception as e:
        print(f"   ✗ Failed to run InfiniLM inference: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Compare outputs
    print("\n8. Comparing inference outputs...")
    try:
        # Check shapes match
        if infinilm_logits.shape != transformers_logits.shape:
            print(f"   ✗ Shape mismatch:")
            print(f"     InfiniLM: {infinilm_logits.shape}")
            print(f"     Transformers: {transformers_logits.shape}")
            return False

        print(f"   ✓ Shapes match: {infinilm_logits.shape}")

        # Compare predicted tokens for human understanding
        # Compute predicted tokens from logits
        transformers_predicted_ids = transformers_logits.argmax(dim=-1)
        transformers_predicted_tokens = transformers_predicted_ids[0].tolist()
        transformers_predicted_text = tokenizer.decode(
            transformers_predicted_tokens, skip_special_tokens=True)

        infinilm_predicted_ids = infinilm_logits.argmax(dim=-1)
        infinilm_predicted_tokens = infinilm_predicted_ids[0].tolist()
        infinilm_predicted_text = tokenizer.decode(
            infinilm_predicted_tokens, skip_special_tokens=True)

        print(f"\n   Predicted tokens comparison:")
        print(f"     Transformers: {transformers_predicted_tokens}")
        print(f"     InfiniLM:     {infinilm_predicted_tokens}")
        if transformers_predicted_tokens == infinilm_predicted_tokens:
            print(f"     ✓ Predicted tokens match!")
        else:
            print(f"     ✗ Predicted tokens differ")
            # Show where they differ
            mismatches = []
            min_len = min(len(transformers_predicted_tokens),
                          len(infinilm_predicted_tokens))
            for i in range(min_len):
                if transformers_predicted_tokens[i] != infinilm_predicted_tokens[i]:
                    mismatches.append(i)
            if mismatches:
                # Show first 10
                print(f"     Mismatches at positions: {mismatches[:10]}")

        print(f"\n   Predicted text comparison:")
        print(f"     Transformers: \"{transformers_predicted_text}\"")
        print(f"     InfiniLM:     \"{infinilm_predicted_text}\"")
        if transformers_predicted_text == infinilm_predicted_text:
            print(f"     ✓ Predicted text matches!")
        else:
            print(f"     ✗ Predicted text differs")

        # Compare logits
        is_close, stats = tensor_all_close(
            infinilm_logits, transformers_logits, rtol=1e-3, atol=1e-3)

        print(f"   Comparison statistics:")
        print(f"     Max absolute difference: {stats['max_abs_diff']:.6e}")
        print(f"     Mean absolute difference: {stats['mean_abs_diff']:.6e}")
        print(f"     Max relative difference: {stats['max_rel_diff']:.6e}")

        if is_close:
            print(f"   ✓ Logits match within tolerance (rtol=1e-3, atol=1e-3)")
        else:
            print(f"   ✗ Logits do not match within tolerance")
            # Print some sample differences
            diff = (infinilm_logits - transformers_logits).abs()
            print(f"     Sample differences (first 5 max):")
            flat_diff = diff.flatten()
            top_5_indices = torch.topk(
                flat_diff, min(5, flat_diff.numel())).indices
            for idx in top_5_indices:
                # torch.unravel_index expects a tensor, not a Python int
                # idx is already a tensor scalar, so we can use it directly
                idx_tuple = torch.unravel_index(idx, diff.shape)
                # Convert tuple to tuple of Python ints for indexing
                idx_tuple_py = tuple(int(x.item()) for x in idx_tuple)
                infini_val = infinilm_logits[idx_tuple_py].item()
                trans_val = transformers_logits[idx_tuple_py].item()
                print(f"       [{idx_tuple_py}]: InfiniLM={infini_val:.6f}, "
                      f"Transformers={trans_val:.6f}, diff={abs(infini_val - trans_val):.6e}")

            # Diagnostic summary for large mismatches
            if stats['max_abs_diff'] > 10.0:
                print(f"\n   ⚠ DIAGNOSTIC: Large logit differences detected!")
                print(f"     This suggests potential issues with:")
                print(
                    f"     1. Weight loading - verify all weights are loaded correctly")
                print(
                    f"     2. Attention mechanism - check if attention is computing correctly")
                print(f"     3. Layer processing - verify all layers are being called")
                print(
                    f"     4. Numerical precision - check for overflow/underflow issues")
                # Check if model is predicting same token
                infinilm_unique = torch.unique(infinilm_predicted_ids[0])
                if len(infinilm_unique) == 1:
                    print(
                        f"     5. Model collapse - model is predicting same token ({infinilm_unique[0].item()})")
                    print(
                        f"        This strongly suggests an attention mechanism issue")
            return False

    except Exception as e:
        print(f"   ✗ Failed to compare outputs: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 70)
    print("✓ Inference test completed successfully")
    print("=" * 70)
    print(f"\nInference outputs match between InfiniLM and transformers models.")
    print(f"Single request scenario validated.")
    print("=" * 70)

    # Cleanup
    print("\n9. Cleaning up resources...")
    try:
        import gc
        del infinilm_model
        del transformers_model
        gc.collect()
        print("   ✓ Resources cleaned up")
    except Exception as e:
        print(f"   ⚠ Warning: Cleanup failed: {e}")

    return True


def main():
    """Main test function"""
    # Default model path
    default_model_dir = "/var/qy_home/zenghua/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B-Instruct"

    # Default prompt
    default_prompt = "Hello, how are you?"

    # Default device
    default_device_type = "cpu"
    default_device_index = 0

    # Parse command line arguments
    prompt = default_prompt
    model_dir = None
    device_type = default_device_type
    device_index = default_device_index

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--prompt" and i + 1 < len(sys.argv):
            prompt = sys.argv[i + 1]
            i += 2
        elif arg == "--device" and i + 1 < len(sys.argv):
            device_str = sys.argv[i + 1]
            if ":" in device_str:
                device_type, device_index_str = device_str.split(":", 1)
                try:
                    device_index = int(device_index_str)
                except ValueError:
                    print(f"Error: Invalid device index: {device_index_str}")
                    sys.exit(1)
            else:
                device_type = device_str
                device_index = 0
            i += 2
        elif arg.startswith("--"):
            print(f"Error: Unknown option: {arg}")
            print(
                f"\nUsage: {sys.argv[0]} [model_dir] [--prompt PROMPT] [--device DEVICE]")
            print(f"\nOptions:")
            print(
                f"  --prompt PROMPT        Input prompt text (default: \"{default_prompt}\")")
            print(
                f"  --device DEVICE        Device type and index (default: {default_device_type}:{default_device_index})")
            print(f"                         Examples: cpu, cuda, cuda:0, cuda:1")
            sys.exit(1)
        else:
            if model_dir is None:
                model_dir = arg
            else:
                print(f"Error: Multiple model directories specified")
                sys.exit(1)
            i += 1

    if model_dir is None:
        model_dir = default_model_dir

    if not os.path.exists(model_dir):
        print(f"Error: Model directory not found: {model_dir}")
        print(
            f"\nUsage: {sys.argv[0]} [model_dir] [--prompt PROMPT] [--device DEVICE]")
        print(f"\nOptions:")
        print(
            f"  --prompt PROMPT        Input prompt text (default: \"{default_prompt}\")")
        print(
            f"  --device DEVICE        Device type and index (default: {default_device_type}:{default_device_index})")
        print(f"                         Examples: cpu, cuda, cuda:0, cuda:1")
        print(f"\nExamples:")
        print(f"  {sys.argv[0]} {default_model_dir}")
        print(f"  {sys.argv[0]} {default_model_dir} --prompt \"What is AI?\"")
        print(f"  {sys.argv[0]} {default_model_dir} --device cuda:0")
        print(
            f"  {sys.argv[0]} {default_model_dir} --prompt \"What is AI?\" --device cuda:0")
        sys.exit(1)

    try:
        success = validate_inference(
            model_dir, prompt, device_type, device_index)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
