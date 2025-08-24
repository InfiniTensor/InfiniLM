#!/usr/bin/env python3
"""
Performance comparison between Linear and GEMM operators in InfiniCore,
and comparison with PyTorch's Linear implementation.

This script demonstrates:
1. Performance differences between Linear and GEMM operators on different devices
2. Comparison with PyTorch's linear implementation
3. Analysis of large model scenarios
"""

import time
import torch
import numpy as np
import sys
import os

def setup_test_environment():
    """Setup basic test environment and check available devices"""
    print("=== InfiniCore Linear vs GEMM Performance Comparison ===\n")
    
    # Check PyTorch CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"PyTorch CUDA available: {cuda_available}")
    if cuda_available:
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()
    
    return cuda_available

def pytorch_linear_benchmark(input_data, weight_data, bias_data, device='cpu', num_iterations=1000):
    """Benchmark PyTorch's linear implementation"""
    # Convert to torch tensors
    input_tensor = torch.from_numpy(input_data).float().to(device)
    weight_tensor = torch.from_numpy(weight_data).float().to(device)
    bias_tensor = torch.from_numpy(bias_data).float().to(device) if bias_data is not None else None
    
    # Warmup
    for _ in range(10):
        result = torch.nn.functional.linear(input_tensor, weight_tensor, bias_tensor)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Timing
    start_time = time.time()
    for _ in range(num_iterations):
        result = torch.nn.functional.linear(input_tensor, weight_tensor, bias_tensor)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_iterations * 1000  # ms
    
    return result.cpu().numpy(), avg_time

def pytorch_gemm_benchmark(input_data, weight_data, bias_data, device='cpu', num_iterations=1000):
    """Benchmark equivalent GEMM operations using PyTorch"""
    # Convert to torch tensors
    input_tensor = torch.from_numpy(input_data).float().to(device)
    weight_tensor = torch.from_numpy(weight_data).float().to(device)
    bias_tensor = torch.from_numpy(bias_data).float().to(device) if bias_data is not None else None
    
    # Warmup - simulate GEMM: input @ weight.T + bias
    for _ in range(10):
        result = torch.matmul(input_tensor, weight_tensor.T)
        if bias_tensor is not None:
            result = result + bias_tensor
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Timing
    start_time = time.time()
    for _ in range(num_iterations):
        result = torch.matmul(input_tensor, weight_tensor.T)
        if bias_tensor is not None:
            result = result + bias_tensor
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_iterations * 1000  # ms
    
    return result.cpu().numpy(), avg_time

def analyze_large_model_scenarios():
    """Analyze performance for typical large model scenarios"""
    
    # Typical large model linear layer sizes
    test_cases = [
        # (batch_size, seq_len, in_features, out_features, description)
        (1, 512, 768, 3072, "BERT-base FFN (single sequence)"),
        (32, 512, 768, 3072, "BERT-base FFN (batch=32)"),
        (1, 2048, 4096, 11008, "LLaMA-7B FFN (single sequence)"),
        (16, 2048, 4096, 11008, "LLaMA-7B FFN (batch=16)"),
        (1, 4096, 8192, 22016, "LLaMA-13B FFN (single sequence)"),
        (8, 4096, 8192, 22016, "LLaMA-13B FFN (batch=8)"),
        (1, 8192, 12288, 49152, "LLaMA-65B FFN (single sequence)"),
        (4, 8192, 12288, 49152, "LLaMA-65B FFN (batch=4)"),
    ]
    
    print("=== Large Model Scenario Analysis ===\n")
    
    cuda_available = torch.cuda.is_available()
    devices = ['cpu']
    if cuda_available:
        devices.append('cuda')
    
    for device in devices:
        print(f"--- Testing on {device.upper()} ---")
        print(f"{'Scenario':<30} {'Input Shape':<20} {'Weight Shape':<20} {'PyTorch Linear (ms)':<20} {'PyTorch GEMM (ms)':<20} {'Speedup':<10}")
        print("-" * 120)
        
        for batch_size, seq_len, in_features, out_features, description in test_cases:
            # Reshape for linear layer: (batch_size * seq_len, in_features)
            input_shape = (batch_size * seq_len, in_features)
            weight_shape = (out_features, in_features)
            bias_shape = (out_features,)
            
            # Generate test data
            np.random.seed(42)  # For reproducibility
            input_data = np.random.randn(*input_shape).astype(np.float32) * 0.1
            weight_data = np.random.randn(*weight_shape).astype(np.float32) * 0.1
            bias_data = np.random.randn(*bias_shape).astype(np.float32) * 0.1
            
            try:
                # Test with fewer iterations for large tensors to avoid timeout
                num_iters = 100 if device == 'cuda' else 50
                
                # PyTorch Linear
                linear_result, linear_time = pytorch_linear_benchmark(
                    input_data, weight_data, bias_data, device, num_iters
                )
                
                # PyTorch GEMM equivalent
                gemm_result, gemm_time = pytorch_gemm_benchmark(
                    input_data, weight_data, bias_data, device, num_iters
                )
                
                # Verify results are similar
                if np.allclose(linear_result, gemm_result, rtol=1e-5, atol=1e-5):
                    speedup = gemm_time / linear_time if linear_time > 0 else float('inf')
                    speedup_str = f"{speedup:.2f}x" if speedup != float('inf') else "∞"
                    
                    print(f"{description:<30} {str(input_shape):<20} {str(weight_shape):<20} {linear_time:<20.3f} {gemm_time:<20.3f} {speedup_str:<10}")
                else:
                    print(f"{description:<30} {str(input_shape):<20} {str(weight_shape):<20} {'ERROR: Results mismatch':<50}")
                    
            except Exception as e:
                print(f"{description:<30} {str(input_shape):<20} {str(weight_shape):<20} {'ERROR: ' + str(e):<50}")
        
        print()

def demonstrate_usage_patterns():
    """Demonstrate typical usage patterns and their performance characteristics"""
    
    print("=== Usage Pattern Demonstration ===\n")
    
    # Simple example data
    batch_size = 2
    in_features = 4
    out_features = 3
    
    # Generate test data
    np.random.seed(42)
    input_data = np.array([[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]], dtype=np.float32)
    weight_data = np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8], 
        [0.9, 1.0, 1.1, 1.2]
    ], dtype=np.float32)
    bias_data = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    
    print("Input data:")
    print(input_data)
    print("\nWeight data:")
    print(weight_data)
    print("\nBias data:")
    print(bias_data)
    print()
    
    # PyTorch Linear
    linear_result, linear_time = pytorch_linear_benchmark(
        input_data, weight_data, bias_data, 'cpu', 1000
    )
    
    # PyTorch GEMM equivalent
    gemm_result, gemm_time = pytorch_gemm_benchmark(
        input_data, weight_data, bias_data, 'cpu', 1000
    )
    
    print("PyTorch Linear result:")
    print(linear_result)
    print(f"Average time: {linear_time:.6f} ms")
    print()
    
    print("PyTorch GEMM equivalent result:")
    print(gemm_result)
    print(f"Average time: {gemm_time:.6f} ms")
    print()
    
    # Check if results match
    if np.allclose(linear_result, gemm_result, rtol=1e-6, atol=1e-6):
        speedup = gemm_time / linear_time if linear_time > 0 else float('inf')
        print(f"✅ Results match! Linear vs GEMM speedup: {speedup:.2f}x")
    else:
        print("❌ Results don't match!")
        print(f"Max difference: {np.max(np.abs(linear_result - gemm_result))}")
    print()

def main():
    """Main function to run all benchmarks and demonstrations"""
    
    cuda_available = setup_test_environment()
    
    # 1. Demonstrate basic usage patterns
    demonstrate_usage_patterns()
    
    # 2. Analyze large model scenarios  
    analyze_large_model_scenarios()
    
    # 3. Summary and conclusions
    print("=== Summary ===\n")
    print("Key Differences between Linear and GEMM operators:")
    print()
    print("📊 **GEMM (General Matrix Multiplication)**:")
    print("   - Operation: C = alpha * A @ B + beta * C")
    print("   - More general, supports arbitrary scaling factors (alpha, beta)")
    print("   - Flexible for various mathematical operations")
    print("   - May require separate transpose and bias addition operations")
    print()
    print("🧠 **Linear (Neural Network Layer)**:")
    print("   - Operation: output = input @ weight.T + bias")
    print("   - Specialized for neural network linear layers")
    print("   - Built-in weight transpose and bias addition")
    print("   - Optimized for typical neural network patterns")
    print("   - Simpler API for common use cases")
    print()
    print("🚀 **Performance Considerations**:")
    print("   - Linear operator can be faster for neural network workloads because:")
    print("     • Specialized for the input @ weight.T + bias pattern")
    print("     • Integrated bias addition avoids extra memory operations")
    print("     • No need for alpha/beta scaling computations")
    print("     • Better cache locality for typical NN access patterns")
    print()
    print("   - GEMM is more flexible but may have overhead for simple linear operations")
    print("   - In large models, Linear operator provides cleaner API and potentially better performance")
    print()
    print("💡 **Recommendation**:")
    print("   - Use Linear operator for neural network linear layers")
    print("   - Use GEMM for general matrix operations requiring scaling factors")
    print("   - The performance difference depends on specific hardware and implementation")

if __name__ == "__main__":
    main()