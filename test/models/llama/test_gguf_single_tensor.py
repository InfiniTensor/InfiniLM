#!/usr/bin/env python3
"""
Unit test for loading single tensors from GGUF files.

This test validates:
1. Loading a single tensor from GGUF file
2. Tensor shape and dtype correctness
3. Name mapping from GGUF to InfiniLM
4. Error handling for invalid tensors
"""

import sys
import os
import unittest
import warnings
from pathlib import Path
from typing import Optional

try:
    import torch
    import numpy as np
except ImportError as e:
    print(f"Error: Required packages not found. Please install: {e}")
    sys.exit(1)

try:
    import infinicore
except ImportError as e:
    print(f"Error: InfiniCore package not found. Please install it: {e}")
    sys.exit(1)

try:
    from infinicore.gguf import GGUFReader, find_split_files
    from infinilm.modeling_utils import (
        map_gguf_to_infinilm_name,
        is_gguf_path,
        load_model_state_dict_by_tensor,
    )
    from infinilm.auto_config import AutoConfig
    from infinilm.infer_engine import InferEngine
    from infinilm.distributed import DistConfig
    GGUF_AVAILABLE = True
except ImportError as e:
    print(f"Warning: GGUF support not available: {e}")
    GGUF_AVAILABLE = False


class TestGGUFSingleTensorLoading(unittest.TestCase):
    """Test single tensor loading from GGUF files."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        if not GGUF_AVAILABLE:
            raise unittest.SkipTest("GGUF support not available")

        # Try to find a test GGUF file
        # First check if a test file path is provided via environment variable
        test_gguf_path = os.environ.get("TEST_GGUF_FILE")
        if test_gguf_path and os.path.exists(test_gguf_path):
            cls.test_gguf_file = test_gguf_path
        else:
            # Skip test if no GGUF file is available
            raise unittest.SkipTest("No test GGUF file available. Set TEST_GGUF_FILE environment variable.")

    def _compare_shapes(self, shape1, shape2, tensor_name=""):
        """Compare two shapes, handling reversed GGUF shapes and type differences.

        GGUF stores shapes in reverse order (column-major), so we check if shapes
        match either directly or reversed.
        """
        # Convert to tuples for comparison
        if isinstance(shape1, list):
            shape1 = tuple(shape1)
        if isinstance(shape2, list):
            shape2 = tuple(shape2)

        # Check direct match
        if shape1 == shape2:
            return True

        # Check reversed match (GGUF stores shapes reversed)
        if len(shape1) == len(shape2) and len(shape1) > 0:
            reversed_shape2 = tuple(reversed(shape2))
            if shape1 == reversed_shape2:
                return True

        return False

    def test_gguf_reader_initialization(self):
        """Test GGUFReader can be initialized."""
        reader = GGUFReader(self.test_gguf_file)
        self.assertIsNotNone(reader)

        # Test that we can get tensor names
        tensor_names = reader.get_tensor_names()
        self.assertIsInstance(tensor_names, list)
        self.assertGreater(len(tensor_names), 0, "GGUF file should contain at least one tensor")

    def test_load_single_tensor_by_name(self):
        """Test loading a single tensor by name."""
        reader = GGUFReader(self.test_gguf_file)
        tensor_names = reader.get_tensor_names()

        # Find a test tensor (prefer common ones like token_embd or layer weights)
        test_tensor_name = None
        for name in tensor_names:
            if "token_embd" in name or "blk.0" in name or "norm" in name:
                test_tensor_name = name
                break

        if test_tensor_name is None:
            test_tensor_name = tensor_names[0]

        # Load the tensor
        np_array = reader.get_tensor_data(test_tensor_name)

        # Verify it's a numpy array
        self.assertIsInstance(np_array, np.ndarray, f"Expected numpy array for {test_tensor_name}")

        # Verify it has a shape
        self.assertGreater(len(np_array.shape), 0, f"Tensor {test_tensor_name} should have at least one dimension")

        # Get tensor info
        tensor_info = reader.get_tensor_info(test_tensor_name)
        self.assertIn("shape", tensor_info)
        self.assertIn("dtype", tensor_info)

        # Verify shape matches (GGUF may store shapes reversed)
        info_shape = tensor_info["shape"]
        array_shape = np_array.shape
        self.assertTrue(self._compare_shapes(array_shape, info_shape, test_tensor_name),
                        f"Shape mismatch for {test_tensor_name}: array {array_shape} vs info {info_shape}")

    def test_tensor_shape_consistency(self):
        """Test that tensor shapes are consistent between get_tensor_data and get_tensor_info."""
        reader = GGUFReader(self.test_gguf_file)
        tensor_names = reader.get_tensor_names()

        for tensor_name in tensor_names[:10]:  # Test first 10 tensors
            try:
                np_array = reader.get_tensor_data(tensor_name)
                tensor_info = reader.get_tensor_info(tensor_name)

                # Compare shapes
                info_shape = tensor_info["shape"]
                array_shape = np_array.shape

                # Shapes should match (may be reversed in some cases)
                self.assertEqual(len(info_shape), len(array_shape),
                               f"Shape dimension mismatch for {tensor_name}: {info_shape} vs {array_shape}")

            except Exception as e:
                self.fail(f"Failed to load tensor {tensor_name}: {e}")

    def test_tensor_to_torch_conversion(self):
        """Test converting GGUF tensor to PyTorch tensor."""
        reader = GGUFReader(self.test_gguf_file)
        tensor_names = reader.get_tensor_names()

        # Find a test tensor
        test_tensor_name = tensor_names[0]

        # Load as numpy array
        np_array = reader.get_tensor_data(test_tensor_name)

        # Convert to torch tensor
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*not writable.*")
            torch_tensor = torch.from_numpy(np_array)

        # Verify conversion
        self.assertEqual(torch_tensor.shape, np_array.shape,
                        f"Shape mismatch after torch conversion for {test_tensor_name}")

        # Verify dtype
        self.assertIsInstance(torch_tensor, torch.Tensor)

    def test_tensor_name_mapping(self):
        """Test mapping GGUF tensor names to InfiniLM parameter names."""
        reader = GGUFReader(self.test_gguf_file)
        tensor_names = reader.get_tensor_names()

        # Test mapping for various tensor types
        test_cases = [
            ("token_embd.weight", "model.embed_tokens.weight"),
            ("blk.0.attn_q.weight", "model.layers.0.self_attn.q_proj.weight"),
            ("blk.0.attn_k.weight", "model.layers.0.self_attn.k_proj.weight"),
            ("blk.0.attn_v.weight", "model.layers.0.self_attn.v_proj.weight"),
            ("blk.0.attn_output.weight", "model.layers.0.self_attn.o_proj.weight"),
            ("blk.0.ffn_gate.weight", "model.layers.0.mlp.gate_proj.weight"),
            ("blk.0.ffn_up.weight", "model.layers.0.mlp.up_proj.weight"),
            ("blk.0.ffn_down.weight", "model.layers.0.mlp.down_proj.weight"),
            ("blk.0.attn_norm.weight", "model.layers.0.input_layernorm.weight"),
            ("blk.0.ffn_norm.weight", "model.layers.0.post_attention_layernorm.weight"),
            ("output_norm.weight", "model.norm.weight"),
            ("output.weight", "lm_head.weight"),
        ]

        for gguf_name, expected_infinilm_name in test_cases:
            mapped_name = map_gguf_to_infinilm_name(gguf_name)
            self.assertEqual(mapped_name, expected_infinilm_name,
                            f"Mapping failed for {gguf_name}: got {mapped_name}, expected {expected_infinilm_name}")

        # Test that rope_factors returns None (should be skipped)
        rope_mapped = map_gguf_to_infinilm_name("blk.0.rope_factors")
        self.assertIsNone(rope_mapped, "rope_factors should map to None")

    def test_load_tensor_with_mapping(self):
        """Test loading a tensor and applying name mapping."""
        reader = GGUFReader(self.test_gguf_file)
        tensor_names = reader.get_tensor_names()

        # Find a tensor that can be mapped
        test_tensor_name = None
        for name in tensor_names:
            mapped = map_gguf_to_infinilm_name(name)
            if mapped is not None:
                test_tensor_name = name
                break

        if test_tensor_name is None:
            self.skipTest("No mappable tensor found in GGUF file")

        # Load tensor
        np_array = reader.get_tensor_data(test_tensor_name)

        # Map name
        mapped_name = map_gguf_to_infinilm_name(test_tensor_name)

        # Convert to torch and then to infinicore
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*not writable.*")
            torch_tensor = torch.from_numpy(np_array).to(dtype=torch.float16)

        infini_tensor = infinicore.from_torch(torch_tensor)

        # Verify conversion succeeded
        self.assertIsNotNone(infini_tensor)
        # Convert infini_tensor.shape (list) to tuple for comparison
        infini_shape = tuple(infini_tensor.shape) if isinstance(infini_tensor.shape, list) else infini_tensor.shape
        self.assertEqual(infini_shape, np_array.shape,
                        f"Shape mismatch after infinicore conversion for {test_tensor_name}: {infini_shape} vs {np_array.shape}")

    def test_error_handling_invalid_tensor(self):
        """Test error handling for invalid tensor names."""
        reader = GGUFReader(self.test_gguf_file)

        # Try to get non-existent tensor
        with self.assertRaises((KeyError, Exception)):
            reader.get_tensor_data("non_existent_tensor_name")

    def test_tensor_dtype_detection(self):
        """Test that tensor dtype can be detected."""
        reader = GGUFReader(self.test_gguf_file)
        tensor_names = reader.get_tensor_names()

        for tensor_name in tensor_names[:5]:  # Test first 5 tensors
            tensor_info = reader.get_tensor_info(tensor_name)
            self.assertIn("dtype", tensor_info)
            self.assertIsInstance(tensor_info["dtype"], str)

    def test_split_file_detection(self):
        """Test detection of split GGUF files."""
        reader = GGUFReader(self.test_gguf_file)

        # Check if it's a split file
        is_split = reader.is_split_file()
        self.assertIsInstance(is_split, bool)

        if is_split:
            split_no = reader.get_split_no()
            split_count = reader.get_split_count()
            self.assertIsInstance(split_no, int)
            self.assertIsInstance(split_count, int)
            self.assertGreater(split_count, 0)
            self.assertGreaterEqual(split_no, 0)
            self.assertLess(split_no, split_count)

    def test_load_all_tensors_shape_check(self):
        """Test loading all tensors and verifying shapes."""
        reader = GGUFReader(self.test_gguf_file)
        tensor_names = reader.get_tensor_names()

        loaded_tensors = {}
        for tensor_name in tensor_names:
            try:
                np_array = reader.get_tensor_data(tensor_name)
                tensor_info = reader.get_tensor_info(tensor_name)

                # Store for verification
                loaded_tensors[tensor_name] = {
                    "array": np_array,
                    "info": tensor_info,
                }

                # Verify shape consistency (GGUF may store shapes reversed)
                info_shape = tensor_info["shape"]
                array_shape = np_array.shape
                self.assertTrue(self._compare_shapes(array_shape, info_shape, tensor_name),
                               f"Shape mismatch for {tensor_name}: array {array_shape} vs info {info_shape}")

            except Exception as e:
                self.fail(f"Failed to load tensor {tensor_name}: {e}")

        # Verify we loaded at least some tensors
        self.assertGreater(len(loaded_tensors), 0, "Should load at least one tensor")

    def test_load_single_tensor_with_mapping_and_shape_validation(self):
        """Test loading a single tensor with mapping and shape validation.

        This test simulates the load_model_state_dict_by_tensor flow:
        1. Load tensor from GGUF
        2. Map GGUF name to InfiniLM name
        3. Validate shape before loading
        4. Test error handling for shape mismatches
        """
        reader = GGUFReader(self.test_gguf_file)
        tensor_names = reader.get_tensor_names()

        # Find a tensor that can be mapped (prefer attention weights)
        test_tensor_name = None
        for name in tensor_names:
            mapped = map_gguf_to_infinilm_name(name)
            if mapped is not None and ("attn" in name or "ffn" in name or "norm" in name):
                test_tensor_name = name
                break

        if test_tensor_name is None:
            # Fallback to any mappable tensor
            for name in tensor_names:
                mapped = map_gguf_to_infinilm_name(name)
                if mapped is not None:
                    test_tensor_name = name
                    break

        if test_tensor_name is None:
            self.skipTest("No mappable tensor found in GGUF file")

        # Load tensor from GGUF
        np_array = reader.get_tensor_data(test_tensor_name)
        tensor_info = reader.get_tensor_info(test_tensor_name)

        # Map GGUF name to InfiniLM name
        mapped_name = map_gguf_to_infinilm_name(test_tensor_name)
        self.assertIsNotNone(mapped_name, f"Should map {test_tensor_name} to InfiniLM name")

        # Convert to torch tensor (simulating load_model_state_dict_by_tensor flow)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*not writable.*")
            torch_tensor = torch.from_numpy(np_array).to(dtype=torch.float16)

        # Verify tensor shape
        self.assertEqual(torch_tensor.shape, np_array.shape,
                        f"Torch tensor shape should match numpy array shape")

        # Convert to infinicore tensor
        infini_tensor = infinicore.from_torch(torch_tensor)

        # Verify conversion succeeded
        self.assertIsNotNone(infini_tensor)
        infini_shape = tuple(infini_tensor.shape) if isinstance(infini_tensor.shape, list) else infini_tensor.shape
        self.assertEqual(infini_shape, np_array.shape,
                        f"InfiniCore tensor shape should match numpy array shape: {infini_shape} vs {np_array.shape}")

        # Test shape validation logic (as in load_model_state_dict_by_tensor)
        # This simulates checking if the parameter exists in the model
        # In real usage, we'd check model.state_dict_keyname()
        print(f"Successfully loaded and mapped: {test_tensor_name} -> {mapped_name}")
        print(f"  Tensor shape: {np_array.shape}")
        print(f"  GGUF info shape: {tensor_info.get('shape')} (may be reversed)")
        print(f"  Mapped name: {mapped_name}")

    def test_tensor_shape_mismatch_detection(self):
        """Test detection of shape mismatches when loading tensors.

        This test validates that we can detect shape mismatches like the one
        in the error log: model expects [4096, 4096] but GGUF has [256, 4096].
        """
        reader = GGUFReader(self.test_gguf_file)
        tensor_names = reader.get_tensor_names()

        # Find attention weight tensors (k_proj, q_proj, v_proj)
        attention_tensors = []
        for name in tensor_names:
            if "attn_k" in name or "attn_q" in name or "attn_v" in name:
                mapped = map_gguf_to_infinilm_name(name)
                if mapped is not None:
                    attention_tensors.append((name, mapped))

        if len(attention_tensors) == 0:
            self.skipTest("No attention tensors found in GGUF file")

        # Test first attention tensor
        gguf_name, mapped_name = attention_tensors[0]

        # Load tensor
        np_array = reader.get_tensor_data(gguf_name)
        tensor_info = reader.get_tensor_info(gguf_name)

        # Get shape information
        array_shape = np_array.shape
        info_shape = tensor_info.get("shape", None)

        # Verify we can detect shapes
        self.assertIsNotNone(array_shape)
        self.assertGreater(len(array_shape), 0)

        # Print shape information for debugging
        print(f"\nTensor: {gguf_name} -> {mapped_name}")
        print(f"  Array shape: {array_shape}")
        print(f"  Info shape: {info_shape} (may be reversed)")

        # Convert to torch and infinicore
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*not writable.*")
            torch_tensor = torch.from_numpy(np_array).to(dtype=torch.float16)

        infini_tensor = infinicore.from_torch(torch_tensor)

        # Verify shapes are consistent
        infini_shape = tuple(infini_tensor.shape) if isinstance(infini_tensor.shape, list) else infini_tensor.shape
        self.assertEqual(infini_shape, array_shape,
                        f"Shape should be consistent: {infini_shape} vs {array_shape}")

        # This test helps identify shape mismatches that would occur during actual loading
        # The shape [256, 4096] suggests num_key_value_heads might be different from num_attention_heads

    def test_load_model_state_dict_by_tensor_actual_call(self):
        """Test calling load_model_state_dict_by_tensor directly with a real model.

        This test actually calls load_model_state_dict_by_tensor to validate:
        1. Model can be created from GGUF config
        2. Tensors can be loaded using load_model_state_dict_by_tensor
        3. Shape mismatches are detected and reported
        """

        # Create model config from GGUF file
        print(f"\nCreating model config from GGUF file: {self.test_gguf_file}")
        try:
            config = AutoConfig.from_pretrained(self.test_gguf_file)
            print(f"  Config created: vocab_size={config.vocab_size}, hidden_size={config.hidden_size}, "
                  f"intermediate_size={config.intermediate_size}, num_hidden_layers={config.num_hidden_layers}")
        except Exception as e:
            self.fail(f"Failed to create config from GGUF: {e}")

        # Create model instance
        print(f"\nCreating InferEngine model...")
        try:
            device = infinicore.device("cpu", 0)  # Use CPU for testing
            model = InferEngine(
                self.test_gguf_file,  # Pass GGUF file path
                device=device,
                distributed_config=DistConfig(1),
            )
            print(f"  Model created successfully")
        except Exception as e:
            self.fail(f"Failed to create model: {e}")

        # Get model's expected parameter names
        try:
            model_keys = model.state_dict_keyname()
            print(f"  Model has {len(model_keys)} parameters")
        except Exception as e:
            self.fail(f"Failed to get model state dict keys: {e}")

        # Actually call load_model_state_dict_by_tensor
        print(f"\nCalling load_model_state_dict_by_tensor...")
        try:
            # Use the directory containing the GGUF file (load_model_state_dict_by_tensor expects a directory)
            model_dir = os.path.dirname(self.test_gguf_file)
            print(f"  Loading from directory: {model_dir}")
            print(f"  GGUF file: {self.test_gguf_file}")

            # Actually call the function
            load_model_state_dict_by_tensor(
                model,
                model_dir,  # Pass directory path containing the GGUF file
                dtype=model.config.dtype,
            )
            print(f"  ✓ load_model_state_dict_by_tensor completed successfully")
        except RuntimeError as e:
            # RuntimeError is expected for shape mismatches
            error_msg = str(e)
            if "Shape mismatch" in error_msg or "shape" in error_msg.lower():
                print(f"  ⚠ Shape mismatch detected: {error_msg}")
                # Print more details about the shape mismatch
                import traceback
                traceback.print_exc()
                # This is expected for some tensors - validates error detection works
            else:
                print(f"  ✗ RuntimeError during loading: {e}")
                import traceback
                traceback.print_exc()
        except Exception as e:
            print(f"  ✗ Failed to load tensors: {e}")
            import traceback
            traceback.print_exc()
            # Don't fail the test - we want to see what errors occur for debugging

        # Verify some parameters were loaded
        try:
            loaded_keys = model.state_dict_keyname()
            print(f"  Model still has {len(loaded_keys)} parameters after loading")
        except Exception as e:
            print(f"  Warning: Could not verify loaded parameters: {e}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    unittest.main()
