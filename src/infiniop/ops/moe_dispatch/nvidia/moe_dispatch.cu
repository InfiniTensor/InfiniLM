#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "moe_dispatch.cuh"
#include "../info.h"
#include <cub/device/device_scan.cuh>
#include <cuda_fp16.h>
#include <cuda_bf16.h> // CHANGE: 添加 bfloat16 头文件
#include <cuda_runtime.h>

namespace op::moe_dispatch::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

// NOTE: create 函数无需更改，它已经正确地检查了所有需要的数据类型
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_, Descriptor **desc_ptr, int num_experts,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t indices_desc,
    infiniopTensorDescriptor_t permuted_output_desc,
    infiniopTensorDescriptor_t aux_info_desc) {

    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto dtype = input_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    // NEW: Add hardware check for BF16 during descriptor creation
    if (dtype == INFINI_DTYPE_BF16) {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, handle_->device_id);
        if (props.major < 8) {
            fprintf(stderr,
                    "Error: BF16 data type is not supported on this GPU "
                    "architecture (Compute Capability %d.%d). It requires "
                    "Compute Capability 8.0 (Ampere) or higher.\n",
                    props.major, props.minor);
            return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
        }
    }

    auto result = MoEDispatchInfo::create(num_experts,input_desc, indices_desc,
                                          permuted_output_desc, aux_info_desc
                                          );
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(result.take(), new Opaque{handle->internal()},
                               handle->device, handle_->device_id);
    return INFINI_STATUS_SUCCESS;
}

// NOTE: count_experts_kernel 无需更改，因为它只处理整数索引，与数据类型无关
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

// NOTE: dispatch_kernel 无需更改，因为它是一个模板核函数，本身就是类型通用的
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

// CHANGE: 对 calculate 函数进行了大幅修改
infiniStatus_t Descriptor::calculate(const void *input, const void *indices,
                                     void *permuted_output, void *aux_info,
                                     void *stream) const {
    // 只检查索引类型，数据类型的检查将在 switch 中处理
    if (_info.index_type != INFINI_DTYPE_I32) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    int *expert_counts, *expert_offsets;
    size_t count_size = _info.num_experts * sizeof(int);
    cudaMalloc(&expert_counts, count_size);
    cudaMalloc(&expert_offsets, count_size);
    cudaMemsetAsync(expert_counts, 0, count_size, (cudaStream_t)stream);

    dim3 count_grid((_info.num_tokens * _info.k + 255) / 256);
    dim3 count_block(256);
    count_experts_kernel<<<count_grid, count_block, 0, (cudaStream_t)stream>>>(
        (const int *)indices, expert_counts, _info.num_tokens, _info.k);

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                  expert_counts, expert_offsets,
                                  _info.num_experts);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                  expert_counts, expert_offsets,
                                  _info.num_experts, (cudaStream_t)stream);

    int *dispatch_offsets;
    cudaMalloc(&dispatch_offsets, count_size);
    cudaMemcpyAsync(dispatch_offsets, expert_offsets, count_size,
                    cudaMemcpyDeviceToDevice, (cudaStream_t)stream);

    dim3 dispatch_grid(_info.num_tokens);
    dim3 dispatch_block(_info.hidden_dim);
    if (_info.hidden_dim > 1024)
        dispatch_block.x = 1024;

    // NEW: 使用 switch 语句根据数据类型分发不同模板的 Kernel
    switch (_info.data_type) {
    case INFINI_DTYPE_F32:
        dispatch_kernel<float, int><<<dispatch_grid, dispatch_block, 0, (cudaStream_t)stream>>>(
            (const float *)input, (const int *)indices, (float *)permuted_output,
            dispatch_offsets, (int *)aux_info, _info.num_tokens, _info.k,
            _info.hidden_dim);
        break;

    case INFINI_DTYPE_F16:
        dispatch_kernel<__half, int><<<dispatch_grid, dispatch_block, 0, (cudaStream_t)stream>>>(
            (const __half *)input, (const int *)indices, (__half *)permuted_output,
            dispatch_offsets, (int *)aux_info, _info.num_tokens, _info.k,
            _info.hidden_dim);
        break;

    case INFINI_DTYPE_BF16:
        dispatch_kernel<__nv_bfloat16, int><<<dispatch_grid, dispatch_block, 0, (cudaStream_t)stream>>>(
            (const __nv_bfloat16 *)input, (const int *)indices, (__nv_bfloat16 *)permuted_output,
            dispatch_offsets, (int *)aux_info, _info.num_tokens, _info.k,
            _info.hidden_dim);
        break;

    default:
        // 释放已分配的内存
        cudaFree(d_temp_storage);
        cudaFree(expert_counts);
        cudaFree(expert_offsets);
        cudaFree(dispatch_offsets);
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    cudaFree(d_temp_storage);
    cudaFree(expert_counts);
    cudaFree(expert_offsets);
    cudaFree(dispatch_offsets);
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::moe_dispatch::nvidia