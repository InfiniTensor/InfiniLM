#include "../../../devices/moore/moore_handle.h"
#include "flipud_moore.h"
#include "flipud_moore_kernel.h"
#include <algorithm>
#include <cstdint>
#include <vector>

namespace op::flipud::moore {

// ==================================================================
// 辅助函数
// ==================================================================
static inline bool is_pointer_aligned(const void *ptr, size_t alignment) {
    return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}

// ==================================================================
// Opaque 定义：存储 Tensor Layout
// ==================================================================
struct Descriptor::Opaque {
    op::flipud::moore::TensorLayout layout;
};

// ==================================================================
// Kernel Launch Logic
// ==================================================================
template <typename T>
void launch_kernel(
    void *output, const void *input,
    op::flipud::moore::TensorLayout layout,
    size_t numel,
    void *stream) {

    auto in_ptr = reinterpret_cast<const T *>(input);
    auto out_ptr = reinterpret_cast<T *>(output);
    auto musa_stream = reinterpret_cast<musaStream_t>(stream);

    constexpr int TotalBytes = 16; // 128-bit
    constexpr int PackSize = TotalBytes / sizeof(T);

    // ------------------------------------------
    // 向量化判定 (Vectorization Check)
    // ------------------------------------------
    // 1. 指针地址对齐
    bool is_ptr_aligned = is_pointer_aligned(output, TotalBytes) && is_pointer_aligned(input, TotalBytes);

    // 2. 元素总数必须是 PackSize 的倍数
    bool is_numel_divisible = (numel % PackSize == 0);

    // 3. 最后一维大小必须是 PackSize 的倍数 (保证 Pack 不会跨行读取)
    bool is_last_dim_aligned = (layout.ndim > 0) && (layout.shape[layout.ndim - 1] % PackSize == 0);

    // 4. 连续性条件：维度 > 1 且 最内层在内存中是连续的 (stride=1)
    bool is_inner_contiguous = (layout.ndim > 1) && (layout.in_strides[layout.ndim - 1] == 1) && (layout.out_strides[layout.ndim - 1] == 1);

    // 5. 步长对齐条件: 除非是最内层维度，否则所有 Stride 都必须是 PackSize 的倍数
    // 这样保证每个 Pack 读取的起始地址都是对齐的
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

        op::flipud::moore::flipud_kernel_vectorized<T, PackSize>
            <<<grid_size, block_size, 0, musa_stream>>>(out_ptr, in_ptr, num_packs, layout);
    } else {
        size_t block_size = 256;
        size_t grid_size = (numel + block_size - 1) / block_size;

        op::flipud::moore::flipud_kernel<T>
            <<<grid_size, block_size, 0, musa_stream>>>(out_ptr, in_ptr, numel, layout);
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
    infiniopHandle_t handle_, Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc, infiniopTensorDescriptor_t input_desc) {

    auto handle = reinterpret_cast<device::moore::Handle *>(handle_);

    auto info_result = FlipudInfo::create(out_desc, input_desc);
    if (!info_result) {
        return info_result.status();
    }

    auto opaque = new Opaque();
    opaque->layout.ndim = static_cast<int>(input_desc->ndim());

    if (opaque->layout.ndim > op::flipud::moore::MAX_DIMS) {
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

    switch (dtype) {
    case INFINI_DTYPE_F16:
        launch_kernel<half>(output, input, _opaque->layout, numel, stream);
        break;
    case INFINI_DTYPE_BF16:
        launch_kernel<__mt_bfloat16>(output, input, _opaque->layout, numel, stream);
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

} // namespace op::flipud::moore
