#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "moe_dispatch.cuh"
#include "../info.h"
#include <cub/device/device_scan.cuh>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace op::moe_dispatch::cuda {

struct Descriptor::Opaque {};

Descriptor::~Descriptor() {}

// Kernel to count how many tokens are assigned to each expert.
__global__ void count_experts_kernel(const int *indices, int *expert_counts,
                                     int num_tokens, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_tokens * k) {
        int expert_idx = indices[i];
        if (expert_idx >= 0) { // Assuming invalid experts are marked as < 0
            atomicAdd(&expert_counts[expert_idx], 1);
        }
    }
}

// The main dispatch kernel that scatters the tokens based on the expert
// offsets.
template <typename T, typename IndT>
__global__ void dispatch_kernel(const T *input, const IndT *indices,
                                T *permuted_output, int *expert_offsets,
                                int *aux_info, int num_tokens, int k,
                                int hidden_dim) {
    int token_idx = blockIdx.x;
    int data_idx = threadIdx.x;

    if (token_idx >= num_tokens || data_idx >= hidden_dim)
        return;

    for (int i = 0; i < k; ++i) {
        int expert_idx = indices[token_idx * k + i];
        if (expert_idx >= 0) {
            int dispatch_pos = atomicAdd(&expert_offsets[expert_idx], 1);
            permuted_output[dispatch_pos * hidden_dim + data_idx] =
                input[token_idx * hidden_dim + data_idx];
            if (data_idx == 0) {
                aux_info[dispatch_pos * 2 + 0] = token_idx;
                aux_info[dispatch_pos * 2 + 1] = token_idx * k + i;
            }
        }
    }
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_, Descriptor **desc_ptr, int num_experts,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t indices_desc,
    infiniopTensorDescriptor_t permuted_output_desc,
    infiniopTensorDescriptor_t aux_info_desc) {

    auto result = MoEDispatchInfo::create(input_desc, indices_desc,
                                          permuted_output_desc, aux_info_desc,
                                          num_experts);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(result.take(), nullptr, INFINI_DEVICE_NVIDIA,
                               handle_->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(const void *input, const void *indices,
                                     void *permuted_output, void *aux_info,
                                     void *stream) const {
    if (_info.data_type() != INFINI_DTYPE_F32 ||
        _info.index_type() != INFINI_DTYPE_I32) {
        IT_TODO_HALT_MSG("Unsupported data type for MoE Dispatch");
    }

    int *expert_counts, *expert_offsets;
    size_t count_size = _info.num_experts() * sizeof(int);
    cudaMalloc(&expert_counts, count_size);
    cudaMalloc(&expert_offsets, count_size);
    cudaMemsetAsync(expert_counts, 0, count_size, (cudaStream_t)stream);

    dim3 count_grid((_info.num_tokens() * _info.k() + 255) / 256);
    dim3 count_block(256);
    count_experts_kernel<<<count_grid, count_block, 0, (cudaStream_t)stream>>>(
        (const int *)indices, expert_counts, _info.num_tokens(), _info.k());

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                  expert_counts, expert_offsets,
                                  _info.num_experts());
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                  expert_counts, expert_offsets,
                                  _info.num_experts(), (cudaStream_t)stream);

    int *dispatch_offsets;
    cudaMalloc(&dispatch_offsets, count_size);
    cudaMemcpyAsync(dispatch_offsets, expert_offsets, count_size,
                    cudaMemcpyDeviceToDevice, (cudaStream_t)stream);

    dim3 dispatch_grid(_info.num_tokens());
    dim3 dispatch_block(_info.hidden_dim());
    if (_info.hidden_dim() > 1024)
        dispatch_block.x = 1024;

    dispatch_kernel<float, int><<<dispatch_grid, dispatch_block, 0,
                                 (cudaStream_t)stream>>>(
        (const float *)input, (const int *)indices, (float *)permuted_output,
        dispatch_offsets, (int *)aux_info, _info.num_tokens(), _info.k(),
        _info.hidden_dim());

    cudaFree(d_temp_storage);
    cudaFree(expert_counts);
    cudaFree(expert_offsets);
    cudaFree(dispatch_offsets);
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::moe_dispatch::cuda