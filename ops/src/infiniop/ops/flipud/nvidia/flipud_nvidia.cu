#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../handle.h"

#include "../cuda/kernel.cuh"
#include "flipud_nvidia.cuh"
#include <algorithm>
#include <cstdint>
#include <vector>

namespace op::flipud::nvidia {

// ==================================================================
// 辅助函数
// ==================================================================
// [修改点 1] 去掉 template <typename T>，改为普通静态函数，避免解析错误
// [修改点 2] 重命名为 is_pointer_aligned 避免潜在的命名冲突
static inline bool is_pointer_aligned(const void *ptr, size_t alignment) {
    return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}

// ==================================================================
// Opaque 定义：存储 Tensor Layout
// ==================================================================
// [关键] 必须在析构函数之前定义完整结构
struct Descriptor::Opaque {
    op::flipud::cuda::TensorLayout layout;
};

// ==================================================================
// Kernel Launch Logic
// ==================================================================
template <typename T>
void launch_kernel(
    void *output, const void *input,
    op::flipud::cuda::TensorLayout layout,
    size_t numel,
    void *stream) {

    auto in_ptr = reinterpret_cast<const T *>(input);
    auto out_ptr = reinterpret_cast<T *>(output);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    constexpr int TotalBytes = 16; // 128-bit
    constexpr int PackSize = TotalBytes / sizeof(T);

    // ------------------------------------------
    // 向量化判定 (Vectorization Check)
    // ------------------------------------------
    bool is_ptr_aligned = is_pointer_aligned(output, TotalBytes) && is_pointer_aligned(input, TotalBytes);

    bool is_numel_divisible = (numel % PackSize == 0);

    bool is_last_dim_aligned = (layout.ndim > 0) && (layout.shape[layout.ndim - 1] % PackSize == 0);

    // 4. 连续性条件：维度 > 1 且 最内层连续
    bool is_inner_contiguous = (layout.ndim > 1) && (layout.in_strides[layout.ndim - 1] == 1) && (layout.out_strides[layout.ndim - 1] == 1);

    // 5. 步长对齐条件
    bool is_stride_aligned = true;
    for (int i = 0; i < layout.ndim - 1; ++i) {
        if (layout.in_strides[i] % PackSize != 0 || layout.out_strides[i] % PackSize != 0) {
            is_stride_aligned = false;
            break;
        }
    }

    bool can_vectorize = (PackSize > 1) && is_ptr_aligned && is_numel_divisible && is_last_dim_aligned && is_inner_contiguous && is_stride_aligned;

    if (can_vectorize) {
        size_t num_packs = numel / PackSize;
        size_t block_size = 256;
        size_t grid_size = (num_packs + block_size - 1) / block_size;

        op::flipud::cuda::flipud_kernel_vectorized<T, PackSize>
            <<<grid_size, block_size, 0, cuda_stream>>>(out_ptr, in_ptr, num_packs, layout);
    } else {
        size_t block_size = 256;
        size_t grid_size = (numel + block_size - 1) / block_size;

        op::flipud::cuda::flipud_kernel<T>
            <<<grid_size, block_size, 0, cuda_stream>>>(out_ptr, in_ptr, numel, layout);
    }
}

// ==================================================================
// Descriptor 实现
// ==================================================================
Descriptor::~Descriptor() {
    if (_opaque) {
        delete _opaque;
    }
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle, Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc, infiniopTensorDescriptor_t input_desc) {

    auto info_result = FlipudInfo::create(out_desc, input_desc);
    if (!info_result) {
        return info_result.status();
    }

    auto opaque = new Opaque();
    opaque->layout.ndim = static_cast<int>(input_desc->ndim());

    if (opaque->layout.ndim > op::flipud::cuda::MAX_DIMS) {
        delete opaque;
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    const auto &shape = input_desc->shape();
    const auto &in_strides = input_desc->strides();
    const auto &out_strides = out_desc->strides();

    for (int i = 0; i < opaque->layout.ndim; ++i) {
        opaque->layout.shape[i] = shape[i];
        opaque->layout.in_strides[i] = in_strides[i];
        opaque->layout.out_strides[i] = out_strides[i];
    }

    *desc_ptr = new Descriptor(opaque, info_result.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size, void *output,
    const void *input, void *stream) const {

    auto dtype = _info.dtype();
    auto numel = _info.numel();

    // 显式 Switch-Case 分发
    switch (dtype) {
    case INFINI_DTYPE_F16:
        launch_kernel<half>(output, input, _opaque->layout, numel, stream);
        break;
    case INFINI_DTYPE_BF16:
        launch_kernel<cuda_bfloat16>(output, input, _opaque->layout, numel, stream);
        break;
    case INFINI_DTYPE_F32:
        launch_kernel<float>(output, input, _opaque->layout, numel, stream);
        break;
    case INFINI_DTYPE_F64:
        launch_kernel<double>(output, input, _opaque->layout, numel, stream);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::flipud::nvidia
