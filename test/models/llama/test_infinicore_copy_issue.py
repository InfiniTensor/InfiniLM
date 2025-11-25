#!/usr/bin/env python3
"""
Minimal test to isolate InfiniCore tensor copy issue on CPU when GPU support is compiled.
This test reproduces the "infiniopRearrange failed" error.
"""

import sys
import torch

try:
    import infinicore
except ImportError as e:
    print(f"Error: InfiniCore package not found: {e}")
    sys.exit(1)


def test_cpu_tensor_copy():
    """Test copying InfiniCore tensors on CPU"""
    print("=" * 70)
    print("Testing InfiniCore Tensor Copy on CPU")
    print("=" * 70)

    # Create a simple PyTorch tensor on CPU
    shape = [128, 256]
    torch_tensor = torch.randn(shape, dtype=torch.float32, device="cpu")
    print(f"\n1. Created PyTorch tensor on CPU")
    print(f"   Shape: {torch_tensor.shape}")
    print(f"   Device: {torch_tensor.device}")

    # Create InfiniCore tensor from PyTorch tensor
    print(f"\n2. Creating InfiniCore tensor from PyTorch tensor...")
    try:
        cpu_device = infinicore.device("cpu", 0)
        infini_tensor = infinicore.from_blob(
            torch_tensor.data_ptr(),
            list(torch_tensor.shape),
            dtype=infinicore.float32,
            device=cpu_device,
        )
        print(f"   ✓ InfiniCore tensor created")
        print(f"   Shape: {infini_tensor.shape}")
        print(f"   Device: {str(infini_tensor.device)}")
        print(f"   Is contiguous: {infini_tensor.is_contiguous()}")
    except Exception as e:
        print(f"   ✗ Failed to create InfiniCore tensor: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Create a destination tensor
    print(f"\n3. Creating destination InfiniCore tensor...")
    try:
        dest_tensor = infinicore.empty(
            list(infini_tensor.shape),
            dtype=infinicore.float32,
            device=cpu_device,
        )
        print(f"   ✓ Destination tensor created")
        print(f"   Shape: {dest_tensor.shape}")
        print(f"   Device: {str(dest_tensor.device)}")
        print(f"   Is contiguous: {dest_tensor.is_contiguous()}")
    except Exception as e:
        print(f"   ✗ Failed to create destination tensor: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Try to make source tensor contiguous
    print(f"\n4. Ensuring source tensor is contiguous...")
    try:
        if not infini_tensor.is_contiguous():
            print(f"   Source tensor is not contiguous, making it contiguous...")
            infini_tensor = infini_tensor.contiguous()
            print(f"   ✓ Made contiguous")
        else:
            print(f"   ✓ Source tensor is already contiguous")
    except Exception as e:
        print(f"   ✗ Failed to make tensor contiguous: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Try copying
    print(f"\n5. Attempting to copy tensor...")
    try:
        print(f"   Source device: {str(infini_tensor.device)}")
        print(f"   Dest device: {str(dest_tensor.device)}")
        print(f"   Source contiguous: {infini_tensor.is_contiguous()}")
        print(f"   Dest contiguous: {dest_tensor.is_contiguous()}")

        dest_tensor.copy_(infini_tensor)
        print(f"   ✓ Copy succeeded!")
        return True
    except Exception as e:
        print(f"   ✗ Copy failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cpu_tensor_copy_with_to():
    """Test copying using .to() method"""
    print("\n" + "=" * 70)
    print("Testing InfiniCore Tensor Copy using .to() method")
    print("=" * 70)

    shape = [128, 256]
    torch_tensor = torch.randn(shape, dtype=torch.float32, device="cpu")

    print(f"\n1. Creating InfiniCore tensor...")
    try:
        cpu_device = infinicore.device("cpu", 0)
        infini_tensor = infinicore.from_blob(
            torch_tensor.data_ptr(),
            list(torch_tensor.shape),
            dtype=infinicore.float32,
            device=cpu_device,
        )
        print(f"   ✓ Created")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False

    print(f"\n2. Using .to() to create a copy...")
    try:
        # Try using .to() to create a copy on the same device
        copied_tensor = infini_tensor.to(cpu_device)
        print(f"   ✓ .to() succeeded")
        print(f"   Original device: {str(infini_tensor.device)}")
        print(f"   Copied device: {str(copied_tensor.device)}")
        return True
    except Exception as e:
        print(f"   ✗ .to() failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cpu_tensor_copy_from_contiguous():
    """Test copying from a contiguous tensor created with empty"""
    print("\n" + "=" * 70)
    print("Testing InfiniCore Tensor Copy from contiguous empty tensor")
    print("=" * 70)

    shape = [128, 256]

    print(f"\n1. Creating source tensor with empty()...")
    try:
        cpu_device = infinicore.device("cpu", 0)
        source_tensor = infinicore.empty(
            shape,
            dtype=infinicore.float32,
            device=cpu_device,
        )
        # Fill with some data using from_blob
        torch_source = torch.randn(shape, dtype=torch.float32, device="cpu")
        source_from_blob = infinicore.from_blob(
            torch_source.data_ptr(),
            shape,
            dtype=infinicore.float32,
            device=cpu_device,
        )
        print(f"   ✓ Created source tensor")
        print(f"   Source contiguous: {source_from_blob.is_contiguous()}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"\n2. Creating destination tensor...")
    try:
        dest_tensor = infinicore.empty(
            shape,
            dtype=infinicore.float32,
            device=cpu_device,
        )
        print(f"   ✓ Created destination tensor")
        print(f"   Dest contiguous: {dest_tensor.is_contiguous()}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False

    print(f"\n3. Attempting copy...")
    try:
        dest_tensor.copy_(source_from_blob)
        print(f"   ✓ Copy succeeded!")
        return True
    except Exception as e:
        print(f"   ✗ Copy failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    results = []

    results.append(("Direct copy", test_cpu_tensor_copy()))
    results.append(("Copy using .to()", test_cpu_tensor_copy_with_to()))
    results.append(("Copy from empty tensor",
                   test_cpu_tensor_copy_from_contiguous()))

    print("\n" + "=" * 70)
    print("Test Results Summary")
    print("=" * 70)
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")

    all_passed = all(result for _, result in results)
    return all_passed


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
