#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../handle.h"

// Iluvatar does not support atomic add yet
#ifndef ENABLE_ILUVATAR_API
#include "../cuda/kernel.cuh"
#endif

#include "index_add_nvidia.cuh"
#include <cstdint>

namespace op::index_add::nvidia {

template <typename T>
bool is_aligned(const void *ptr, size_t alignment) {
    return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}

// ==================================================================
// Kernel Launch Logic
// ==================================================================
template <typename T, typename TIdx>
void launch_kernel(
    void *output,
    const void *source,
    const void *indices,
    const IndexAddInfo &info,
    void *stream) {
// Iluvatar does not support atomic add yet
#ifndef ENABLE_ILUVATAR_API
    auto out_ptr = reinterpret_cast<T *>(output);
    auto src_ptr = reinterpret_cast<const T *>(source);
    auto idx_ptr = reinterpret_cast<const TIdx *>(indices);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    // 获取几何信息
    size_t outer_size = info.outer_size();
    size_t inner_size = info.inner_size();
    size_t dim_size = info.dim_size();
    size_t index_len = info.index_len();
    float alpha = info.alpha();

    // Source 总元素数
    size_t num_source = outer_size * index_len * inner_size;

    // --- 向量化参数配置 ---
    // 目标：每个线程读取 128-bit (16 Bytes) Source 数据
    constexpr int TotalBytes = 16;
    constexpr int PackSize = TotalBytes / sizeof(T);
    bool can_vectorize = (PackSize > 1) && (num_source % PackSize == 0) && is_aligned<T>(source, TotalBytes);

    if (can_vectorize) {
        // === 路径 A: 向量化读取 Kernel ===
        size_t num_packs = num_source / PackSize;

        size_t block_size = 256;
        size_t grid_size = (num_packs + block_size - 1) / block_size;

        op::index_add::cuda::index_add_kernel_vectorized<T, TIdx, PackSize>
            <<<grid_size, block_size, 0, cuda_stream>>>(
                out_ptr, src_ptr, idx_ptr,
                outer_size, inner_size, dim_size, index_len,
                num_packs, alpha);
    } else {
        // === 路径 B: 标量 Kernel ===
        size_t block_size = 256;
        size_t grid_size = (num_source + block_size - 1) / block_size;

        op::index_add::cuda::index_add_kernel<T, TIdx>
            <<<grid_size, block_size, 0, cuda_stream>>>(
                out_ptr, src_ptr, idx_ptr,
                outer_size, inner_size, dim_size, index_len,
                num_source, alpha);
    }
#endif
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
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t in_desc,
    int64_t dim,
    infiniopTensorDescriptor_t index_desc,
    infiniopTensorDescriptor_t source_desc,
    float alpha) {

    // Info 创建
    auto info_result = IndexAddInfo::create(out_desc, in_desc, dim, index_desc, source_desc, alpha);
    if (!info_result) {
        return info_result.status();
    }

    *desc_ptr = new Descriptor(
        new Opaque(), info_result.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

// ==================================================================
// Calculate
// ==================================================================
infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    const void *index,
    const void *source,
    void *stream) const {

    auto dtype = _info.dtype();
    auto idx_dtype = _info.idx_dtype();
#define LAUNCH_BY_SIZE(T_STORAGE)                                                \
    switch (idx_dtype) {                                                         \
    case INFINI_DTYPE_I32:                                                       \
        launch_kernel<T_STORAGE, int32_t>(output, source, index, _info, stream); \
        break;                                                                   \
    case INFINI_DTYPE_I64:                                                       \
        launch_kernel<T_STORAGE, int64_t>(output, source, index, _info, stream); \
        break;                                                                   \
    default:                                                                     \
        return INFINI_STATUS_BAD_TENSOR_DTYPE;                                   \
    }

    switch (dtype) {
    // 32-bit Float
    case INFINI_DTYPE_F32:
        LAUNCH_BY_SIZE(float);
        break;
    // 64-bit Float
    case INFINI_DTYPE_F64:
        LAUNCH_BY_SIZE(double);
        break;
    // 16-bit Half (fp16) -> 使用 __half
    case INFINI_DTYPE_F16:
        LAUNCH_BY_SIZE(__half);
        break;
    // 16-bit BFloat16 (bf16) -> 使用 __nv_bfloat16
    case INFINI_DTYPE_BF16:
        LAUNCH_BY_SIZE(__nv_bfloat16);
        break;
    // Integers
    case INFINI_DTYPE_I32:
        LAUNCH_BY_SIZE(int32_t);
        break;
    case INFINI_DTYPE_I64:
        LAUNCH_BY_SIZE(int64_t);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

#undef LAUNCH_BY_SIZE
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::index_add::nvidia
