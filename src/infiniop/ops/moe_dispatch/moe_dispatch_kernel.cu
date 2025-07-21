#include "moe_dispatch_info.h"
#include "utils/data_type.h"
#include <cuda_fp16.h>
#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>

namespace infiniop {

void MoEDispatchInfo::op(void *permuted_output, void *aux_info, const void *input,
                       const void *indices, cudaStream_t stream) const {
    moe_dispatch_kernel_launcher(input, indices, permuted_output, aux_info,
                                 this->num_tokens, this->k, this->hidden_dim, this->num_experts,
                                 this->data_type, this->index_type, stream);
}

// Kernel to count how many tokens are assigned to each expert.
__global__ void count_experts_kernel(const int *indices, int *expert_counts, int num_tokens, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_tokens * k) {
        int expert_idx = indices[i];
        if (expert_idx >= 0) { // Assuming invalid experts are marked as < 0
            atomicAdd(&expert_counts[expert_idx], 1);
        }
    }
}

// The main dispatch kernel that scatters the tokens based on the expert offsets.
template <typename T, typename IndT>
__global__ void dispatch_kernel(const T *input, const IndT *indices, T *permuted_output,
                                int *expert_offsets, int *aux_info,
                                int num_tokens, int k, int hidden_dim) {
    int token_idx = blockIdx.x;
    int data_idx = threadIdx.x;

    if (token_idx >= num_tokens || data_idx >= hidden_dim) return;

    for (int i = 0; i < k; ++i) {
        int expert_idx = indices[token_idx * k + i];
        if (expert_idx >= 0) {
            // Get the current position for this expert's data and increment it atomically.
            int dispatch_pos = atomicAdd(&expert_offsets[expert_idx], 1);
            
            // Copy the data
            permuted_output[dispatch_pos * hidden_dim + data_idx] = input[token_idx * hidden_dim + data_idx];
            
            // Save info for the combine step: [original_token_pos, gating_val_pos]
            // We only need one thread per token to write this.
            if (data_idx == 0) {
                aux_info[dispatch_pos * 2 + 0] = token_idx;
                aux_info[dispatch_pos * 2 + 1] = token_idx * k + i; // Index to find gating_weight
            }
        }
    }
}


void moe_dispatch_kernel_launcher(const void *input, const void *indices,
                                  void *permuted_output, void *aux_info,
                                  int num_tokens, int k, int hidden_dim, int num_experts,
                                  DataType data_type, DataType index_type,
                                  cudaStream_t stream) {
    if (data_type != DataType::F32 || index_type != DataType::I32) {
        IT_TODO_HALT_MSG("Unsupported data type for MoE Dispatch");
    }

    // Allocate temporary buffers for expert counts and offsets.
    int *expert_counts, *expert_offsets;
    size_t count_size = num_experts * sizeof(int);
    cudaMalloc(&expert_counts, count_size);
    cudaMalloc(&expert_offsets, count_size);
    cudaMemsetAsync(expert_counts, 0, count_size, stream);

    // 1. Count tokens per expert
    dim3 count_grid((num_tokens * k + 255) / 256);
    dim3 count_block(256);
    count_experts_kernel<<<count_grid, count_block, 0, stream>>>(
        (const int*)indices, expert_counts, num_tokens, k);

    // 2. Compute exclusive prefix sum (scan) to get offsets
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, expert_counts, expert_offsets, num_experts);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, expert_counts, expert_offsets, num_experts, stream);
    
    // We need a copy of the offsets for the dispatch kernel, because the kernel modifies it.
    int* dispatch_offsets;
    cudaMalloc(&dispatch_offsets, count_size);
    cudaMemcpyAsync(dispatch_offsets, expert_offsets, count_size, cudaMemcpyDeviceToDevice, stream);

    // 3. Dispatch tokens to their expert slots
    dim3 dispatch_grid(num_tokens);
    dim3 dispatch_block(hidden_dim); // This might be too large, adjust if needed
    if (hidden_dim > 1024) dispatch_block.x = 1024; // Max threads per block

    dispatch_kernel<float, int><<<dispatch_grid, dispatch_block, 0, stream>>>(
        (const float*)input, (const int*)indices, (float*)permuted_output,
        dispatch_offsets, (int*)aux_info,
        num_tokens, k, hidden_dim);

    // Free temporary buffers
    cudaFree(d_temp_storage);
    cudaFree(expert_counts);
    cudaFree(expert_offsets);
    cudaFree(dispatch_offsets);
}

} // namespace infiniop 