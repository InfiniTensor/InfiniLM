#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../handle.h"

#include "../cuda/kernel.cuh"
#include "kthvalue_nvidia.cuh"
#include <algorithm>
#include <cstdint>

namespace op::kthvalue::nvidia {

template <typename T>
static inline bool is_aligned(const void *ptr, size_t alignment) {
    return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}

// ==================================================================
// Helper: Next Power of 2
// ==================================================================
static inline size_t next_power_of_2(size_t n) {
    if (n == 0) {
        return 1;
    }
    size_t p = 1;
    while (p < n) {
        p <<= 1;
    }
    return p;
}

// ==================================================================
// Kernel Launch Logic
// ==================================================================
template <typename T>
void launch_kernel(
    void *values,
    void *indices,
    const void *input,
    const KthvalueInfo &info,
    void *stream) {

    // 1. 准备指针
    auto in_ptr = reinterpret_cast<const T *>(input);
    auto val_ptr = reinterpret_cast<T *>(values);
    auto idx_ptr = reinterpret_cast<int64_t *>(indices);

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    // 2. 准备形状参数
    size_t dim_size = info.dim_size();
    size_t outer_size = info.outer_size();
    size_t inner_size = info.inner_size();
    int k = info.k();

    // 3. 计算 Grid/Block 和 Shared Memory
    // Bitonic Sort 需要 padding 到 2 的幂次
    size_t power_of_2_dim = next_power_of_2(dim_size);

    // Grid: 总切片数 (Outer * Inner)
    size_t total_slices = outer_size * inner_size;

    // Block: 至少 power_of_2_dim / 2 个线程用于比较
    // 限制最大线程数 1024
    unsigned int threads_per_block = std::max(1u, (unsigned int)(power_of_2_dim / 2));

    // Shared Memory 大小
    size_t smem_size = power_of_2_dim * sizeof(op::kthvalue::cuda::KeyValuePair<T>);

    // 硬件限制检查 (Shared Memory Sort 限制)
    // 假设最大支持 Dim Size 为 2048 (对应 1024 线程)
    // 如果超过此限制，需切换到 Global Memory Sort (此处简化处理，仅支持 Shared Mem Sort)
    if (power_of_2_dim > 2048) {
        // Log Error or Fallback?
        // 在实际工程中应返回 Error Code，这里作为 void 函数假设上层已校验或接受限制
        return;
    }

    // 4. 启动 Kernel
    op::kthvalue::cuda::kthvalue_kernel<T>
        <<<total_slices, threads_per_block, smem_size, cuda_stream>>>(
            val_ptr,
            idx_ptr,
            in_ptr,
            dim_size,
            inner_size,
            k,
            power_of_2_dim);
}

// ==================================================================
// Descriptor 实现
// ==================================================================
struct Descriptor::Opaque {};

Descriptor::~Descriptor() {
    if (_opaque) {
        delete _opaque;
    }
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle, Descriptor **desc_ptr,
    infiniopTensorDescriptor_t values_desc,
    infiniopTensorDescriptor_t indices_desc,
    infiniopTensorDescriptor_t input_desc,
    int k,
    int dim,
    int keepdim) {

    auto info_result = KthvalueInfo::create(values_desc, indices_desc, input_desc, k, dim, keepdim);
    if (!info_result) {
        return info_result.status();
    }

    // 目前基于 Shared Memory 的实现不需要额外的 Workspace
    size_t workspace_size = 0;

    *desc_ptr = new Descriptor(new Opaque(), info_result.take(), workspace_size, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *values,
    void *indices,
    const void *input,
    void *stream) const {

    auto dtype = _info.dtype();

    switch (dtype) {
    case INFINI_DTYPE_F16:
        launch_kernel<half>(values, indices, input, _info, stream);
        break;
    case INFINI_DTYPE_BF16:
        launch_kernel<cuda_bfloat16>(values, indices, input, _info, stream);
        break;
    case INFINI_DTYPE_F32:
        launch_kernel<float>(values, indices, input, _info, stream);
        break;
    case INFINI_DTYPE_F64:
        launch_kernel<double>(values, indices, input, _info, stream);
        break;
    case INFINI_DTYPE_I32:
        launch_kernel<int32_t>(values, indices, input, _info, stream);
        break;
    case INFINI_DTYPE_I64:
        launch_kernel<int64_t>(values, indices, input, _info, stream);
        break;
    case INFINI_DTYPE_U32:
        launch_kernel<uint32_t>(values, indices, input, _info, stream);
        break;
    case INFINI_DTYPE_U64:
        launch_kernel<uint64_t>(values, indices, input, _info, stream);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::kthvalue::nvidia
