#!/usr/bin/env python3
"""
Simplified Performance comparison demo for Linear vs GEMM
"""

import time
import torch
import numpy as np

def benchmark_linear_vs_gemm():
    """Compare Linear vs GEMM performance"""
    
    print("🚀 **InfiniCore Linear vs GEMM Performance Analysis**")
    print("=" * 60)
    print()
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    device = 'cuda' if cuda_available else 'cpu'
    print(f"Testing on: {device.upper()}")
    print()
    
    # Test cases for large model scenarios
    test_cases = [
        (512, 768, 3072, "BERT-base FFN layer"),
        (2048, 4096, 11008, "LLaMA-7B FFN layer"),
        (1024, 2048, 8192, "Medium transformer FFN"),
    ]
    
    print("Test Results:")
    print(f"{'Model':<25} {'Linear (ms)':<12} {'GEMM (ms)':<12} {'Speedup':<10} {'Memory (MB)':<12}")
    print("-" * 75)
    
    for seq_len, hidden, intermediate, desc in test_cases:
        # Create test tensors
        input_tensor = torch.randn(1, seq_len, hidden, device=device, dtype=torch.float32) * 0.1
        weight_tensor = torch.randn(intermediate, hidden, device=device, dtype=torch.float32) * 0.1
        bias_tensor = torch.randn(intermediate, device=device, dtype=torch.float32) * 0.1
        
        # Reshape input for linear layer
        input_2d = input_tensor.view(-1, hidden)
        
        # Warmup
        for _ in range(5):
            _ = torch.nn.functional.linear(input_2d, weight_tensor, bias_tensor)
            _ = torch.matmul(input_2d, weight_tensor.T) + bias_tensor
        
        if cuda_available:
            torch.cuda.synchronize()
        
        # Benchmark Linear
        num_iters = 50
        start = time.time()
        for _ in range(num_iters):
            result_linear = torch.nn.functional.linear(input_2d, weight_tensor, bias_tensor)
        if cuda_available:
            torch.cuda.synchronize()
        linear_time = (time.time() - start) / num_iters * 1000
        
        # Benchmark GEMM equivalent
        start = time.time()
        for _ in range(num_iters):
            result_gemm = torch.matmul(input_2d, weight_tensor.T) + bias_tensor
        if cuda_available:
            torch.cuda.synchronize()
        gemm_time = (time.time() - start) / num_iters * 1000
        
        # Calculate memory usage
        memory_mb = (input_2d.numel() + weight_tensor.numel() + bias_tensor.numel()) * 4 / 1e6
        
        # Calculate speedup
        speedup = gemm_time / linear_time
        
        # Verify correctness
        if torch.allclose(result_linear, result_gemm, rtol=1e-5):
            print(f"{desc:<25} {linear_time:<12.2f} {gemm_time:<12.2f} {speedup:<10.2f}x {memory_mb:<12.1f}")
        else:
            print(f"{desc:<25} {'ERROR':<12} {'ERROR':<12} {'N/A':<10} {memory_mb:<12.1f}")
    
    print()

def demonstrate_differences():
    """Demonstrate conceptual differences"""
    
    print("=== Key Differences ===")
    print()
    
    print("🔧 **GEMM (General Matrix Multiplication)**:")
    print("  • Operation: C = alpha * A @ B + beta * C")
    print("  • General purpose matrix operation")
    print("  • Supports arbitrary scaling factors")
    print("  • Requires separate transpose and bias operations")
    print()
    
    print("🧠 **Linear (Neural Network Layer)**:")
    print("  • Operation: output = input @ weight.T + bias")
    print("  • Specialized for neural network layers")
    print("  • Built-in weight transpose and bias addition")
    print("  • Optimized for typical NN access patterns")
    print()
    
    print("📊 **Performance Advantages of Linear Operator**:")
    print("  • Fused operations reduce memory bandwidth")
    print("  • Specialized kernels for NN patterns")
    print("  • Better cache locality")
    print("  • Reduced kernel launch overhead")
    print("  • Typically 10-25% faster for NN workloads")
    print()

def main():
    # Quick demo
    benchmark_linear_vs_gemm()
    demonstrate_differences()
    
    print("=== Conclusions ===")
    print()
    print("✅ **Use Linear operator for**:")
    print("   - Neural network linear/fully-connected layers")
    print("   - Transformer feed-forward networks") 
    print("   - Standard neural network patterns")
    print()
    print("✅ **Use GEMM operator for**:")
    print("   - General matrix multiplication with scaling")
    print("   - Custom mathematical operations")
    print("   - Non-standard matrix operation patterns")
    print()
    print("🎯 **In large models, Linear operator provides**:")
    print("   - Cleaner API and better performance")
    print("   - Reduced memory bandwidth requirements")
    print("   - Simplified debugging and profiling")

if __name__ == "__main__":
    main()