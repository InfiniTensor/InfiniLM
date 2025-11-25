#!/usr/bin/env python3
"""
Test script to validate Llama model skeleton architecture.

This test compares the actual InfiniLM C++ Llama model skeleton with
the transformers model structure:
1. Load model from transformers
2. Create InfiniLM C++ model skeleton
3. Compare state_dict structures
4. Validate parameter shapes match
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, Tuple, List, Set

try:
    import torch
    import transformers
except ImportError as e:
    print(f"Error: Required packages not found. Please install: {e}")
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
from utils import normalize_param_name


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
    # Use the wrapper class which handles dict initialization
    return LlamaConfig(**config_dict)


def compare_state_dicts(transformers_state_dict: Dict[str, torch.Tensor],
                        infinilm_state_dict: Dict[str, Dict],
                        config: dict) -> Tuple[bool, List[str], List[str]]:
    """Compare transformers and InfiniLM state_dicts"""
    errors = []
    warnings = []

    # Normalize transformers keys (remove "model." prefix if present)
    normalized_transformers = {}
    for key, tensor in transformers_state_dict.items():
        normalized_key = normalize_param_name(key)
        normalized_transformers[normalized_key] = tensor

    # Normalize InfiniLM keys (remove "model." prefix if present)
    normalized_infinilm = {}
    for key, param_info in infinilm_state_dict.items():
        normalized_key = normalize_param_name(key)
        normalized_infinilm[normalized_key] = param_info

    # Get normalized keys
    infinilm_keys = set(normalized_infinilm.keys())
    transformers_keys = set(normalized_transformers.keys())

    # Check for missing parameters in InfiniLM
    missing_in_infinilm = transformers_keys - infinilm_keys
    if missing_in_infinilm:
        # Filter out transformers-specific params
        transformers_specific = {'lm_head.bias'}  # Some models have bias
        missing_in_infinilm = missing_in_infinilm - transformers_specific
        if missing_in_infinilm:
            errors.append(
                f"Parameters in transformers but missing in InfiniLM: {', '.join(sorted(missing_in_infinilm)[:10])}")
            if len(missing_in_infinilm) > 10:
                errors.append(f"... and {len(missing_in_infinilm) - 10} more")

    # Check for extra parameters in InfiniLM
    extra_in_infinilm = infinilm_keys - transformers_keys
    if extra_in_infinilm:
        warnings.append(
            f"Parameters in InfiniLM but not in transformers: {', '.join(sorted(extra_in_infinilm)[:10])}")
        if len(extra_in_infinilm) > 10:
            warnings.append(f"... and {len(extra_in_infinilm) - 10} more")

    # Check common parameters for shape mismatches
    common_keys = infinilm_keys & transformers_keys
    shape_mismatches = []

    for key in common_keys:
        infinilm_shape = tuple(normalized_infinilm[key]["shape"])
        transformers_shape = tuple(normalized_transformers[key].shape)

        if infinilm_shape != transformers_shape:
            shape_mismatches.append(
                f"{key}: InfiniLM {infinilm_shape} vs transformers {transformers_shape}")

    if shape_mismatches:
        errors.append("Shape mismatches:")
        for mismatch in shape_mismatches[:20]:
            errors.append(f"  - {mismatch}")
        if len(shape_mismatches) > 20:
            errors.append(f"  ... and {len(shape_mismatches) - 20} more")

    return len(errors) == 0, errors, warnings


def validate_skeleton(model_dir: str) -> bool:
    """Main validation function"""
    print("=" * 70)
    print("Llama Model Skeleton Validation Test")
    print("=" * 70)
    print("\nThis test compares the InfiniLM C++ model skeleton with the")
    print("transformers model structure to ensure compatibility.")
    print("=" * 70)

    # Load configuration
    print("\n1. Loading model configuration...")
    try:
        config_dict = load_model_config(model_dir)
        print(f"   ✓ Configuration loaded from {model_dir}/config.json")
    except Exception as e:
        print(f"   ✗ Failed to load configuration: {e}")
        return False

    # Print configuration
    print("\n   Configuration:")
    print(f"     vocab_size: {config_dict.get('vocab_size')}")
    print(f"     hidden_size: {config_dict.get('hidden_size')}")
    print(f"     intermediate_size: {config_dict.get('intermediate_size')}")
    print(f"     num_hidden_layers: {config_dict.get('num_hidden_layers')}")
    print(
        f"     num_attention_heads: {config_dict.get('num_attention_heads')}")
    num_kv_heads = config_dict.get(
        'num_key_value_heads', config_dict.get('num_attention_heads'))
    if num_kv_heads != config_dict.get('num_attention_heads'):
        print(f"     num_key_value_heads: {num_kv_heads} (GQA)")
    else:
        print(
            f"     num_attention_heads: {config_dict.get('num_attention_heads')} (MHA)")
    print(f"     head_dim: {config_dict.get('head_dim')}")
    print(
        f"     max_position_embeddings: {config_dict.get('max_position_embeddings')}")
    print(f"     rms_norm_eps: {config_dict.get('rms_norm_eps')}")
    print(
        f"     tie_word_embeddings: {config_dict.get('tie_word_embeddings')}")

    # Create InfiniLM config
    print("\n2. Creating InfiniLM LlamaConfig...")
    try:
        infinilm_config = create_llama_config_from_dict(config_dict)
        if not infinilm_config.validate():
            print("   ✗ InfiniLM configuration validation failed")
            return False
        print("   ✓ InfiniLM configuration created and validated")
    except Exception as e:
        print(f"   ✗ Failed to create InfiniLM config: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Create InfiniLM model
    print("\n3. Creating InfiniLM LlamaForCausalLM skeleton...")
    try:
        device = Device()
        infinilm_model = LlamaForCausalLM(infinilm_config, device)
        infinilm_state_dict = infinilm_model.state_dict()
        print(
            f"   ✓ InfiniLM model created with {len(infinilm_state_dict)} parameters")
    except Exception as e:
        print(f"   ✗ Failed to create InfiniLM model: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Load transformers model
    print("\n4. Loading LlamaForCausalLM from transformers...")
    try:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_dir,
            dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        transformers_state_dict = model.state_dict()
        print(
            f"   ✓ Transformers model loaded with {len(transformers_state_dict)} parameters")
    except Exception as e:
        print(f"   ✗ Failed to load transformers model: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Compare skeletons
    print("\n5. Comparing InfiniLM skeleton with transformers model...")
    all_ok, errors, warnings = compare_state_dicts(
        transformers_state_dict, infinilm_state_dict, config_dict
    )

    if warnings:
        print("\n   Warnings:")
        for warning in warnings:
            print(f"     ⚠ {warning}")

    if not all_ok:
        print("\n   ✗ Skeleton comparison failed:")
        for error in errors[:20]:  # Show first 20 errors
            print(f"     - {error}")
        if len(errors) > 20:
            print(f"     ... and {len(errors) - 20} more errors")
        return False

    print("   ✓ InfiniLM skeleton matches transformers model structure")

    # Print summary
    print("\n6. Summary:")
    print(f"   InfiniLM parameters: {len(infinilm_state_dict)}")
    print(f"   Transformers parameters: {len(transformers_state_dict)}")

    # Show some example parameters
    print("\n   Example parameters (first 5):")
    for i, (key, param_info) in enumerate(list(infinilm_state_dict.items())[:5]):
        shape = param_info["shape"]
        print(f"     {key}: {shape}")

    print("\n" + "=" * 70)
    print("✓ All tests PASSED!")
    print("=" * 70)
    print("\nThe InfiniLM C++ skeleton matches the transformers model structure.")
    print("The model can be used with the C++ interface.")
    print("=" * 70)
    return True


def main():
    """Main test function"""
    # Default model path
    default_model_dir = "/var/qy_home/zenghua/.cache/modelscope/hub/models/LLM-Research/Llama-3___2-3B-Instruct"

    # Allow override via command line
    if len(sys.argv) > 1:
        model_dir = sys.argv[1]
    else:
        model_dir = default_model_dir

    if not os.path.exists(model_dir):
        print(f"Error: Model directory not found: {model_dir}")
        print(f"\nUsage: {sys.argv[0]} [model_dir]")
        sys.exit(1)

    try:
        success = validate_skeleton(model_dir)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
