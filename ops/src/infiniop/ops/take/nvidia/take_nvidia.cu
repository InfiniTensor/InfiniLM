#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../handle.h"

#include "../cuda/kernel.cuh"
#include "take_nvidia.cuh"
#include <cstdint>

namespace op::take::nvidia {

// ==================================================================
// 辅助函数：检查指针内存对齐
// ==================================================================
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
    const void *input,
    const void *indices,
    size_t num_out,
    size_t num_in,
    void *stream) {

    auto out_ptr = reinterpret_cast<T *>(output);
    auto in_ptr = reinterpret_cast<const T *>(input);
    auto idx_ptr = reinterpret_cast<const TIdx *>(indices);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    // --- 向量化参数配置 ---
    // 目标：每个线程处理 128-bit (16 Bytes) 数据以最大化带宽
    constexpr int TotalBytes = 16;
    constexpr int PackSize = TotalBytes / sizeof(T);

    // 向量化条件检查：
    // 1. PackSize >= 2: 只有能打包 2 个以上才有意义 (double 是 2 个，float 是 4 个)
    // 2. 整除检查: 输出元素总数必须能被 PackSize 整除 (简化 Kernel 边界判断)
    // 3. 地址对齐: output 指针必须 16 字节对齐
    bool can_vectorize = (PackSize > 1) && (num_out % PackSize == 0) && is_aligned<T>(output, TotalBytes);

    if (can_vectorize) {
        //
        // === 路径 A: 向量化 Kernel (高性能) ===
        size_t num_packs = num_out / PackSize;

        // Block/Grid 配置
        size_t block_size = 256;
        size_t grid_size = (num_packs + block_size - 1) / block_size;

        op::take::cuda::take_kernel_vectorized<T, TIdx, PackSize>
            <<<grid_size, block_size, 0, cuda_stream>>>(
                out_ptr, in_ptr, idx_ptr, num_packs, num_in);
    } else {
        // === 路径 B: 标量 Kernel (回退/兼容) ===
        size_t block_size = 256;
        size_t grid_size = (num_out + block_size - 1) / block_size;

        op::take::cuda::take_kernel<T, TIdx>
            <<<grid_size, block_size, 0, cuda_stream>>>(
                out_ptr, in_ptr, idx_ptr, num_out, num_in);
    }
}

// ==================================================================
// Descriptor 部分保持不变
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
    infiniopTensorDescriptor_t indices_desc) {

    auto info_result = TakeInfo::create(out_desc, in_desc, indices_desc);
    if (!info_result) {
        return info_result.status();
    }

    *desc_ptr = new Descriptor(
        new Opaque(), info_result.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

// ==================================================================
// Calculate: 结合 "按字节分发" 和 "向量化"
// ==================================================================
infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    const void *indices,
    void *stream) const {

    auto dtype = _info.dtype();
    auto idx_dtype = _info.idx_dtype();
    auto num_out = _info.num_out();
    auto num_in = _info.num_in();

// 索引分发宏
#define LAUNCH_BY_SIZE(T_STORAGE)                                                           \
    switch (idx_dtype) {                                                                    \
    case INFINI_DTYPE_I32:                                                                  \
        launch_kernel<T_STORAGE, int32_t>(output, input, indices, num_out, num_in, stream); \
        break;                                                                              \
    case INFINI_DTYPE_I64:                                                                  \
        launch_kernel<T_STORAGE, int64_t>(output, input, indices, num_out, num_in, stream); \
        break;                                                                              \
    default:                                                                                \
        return INFINI_STATUS_BAD_TENSOR_DTYPE;                                              \
    }

    // 根据数据类型字节大小归类
    size_t element_size = 0;
    switch (dtype) {
    case INFINI_DTYPE_I8:
    case INFINI_DTYPE_U8:
        element_size = 1;
        break;
    case INFINI_DTYPE_F16:
    case INFINI_DTYPE_BF16:
    case INFINI_DTYPE_I16:
    case INFINI_DTYPE_U16:
        element_size = 2;
        break;
    case INFINI_DTYPE_F32:
    case INFINI_DTYPE_I32:
    case INFINI_DTYPE_U32:
        element_size = 4;
        break;
    case INFINI_DTYPE_F64:
    case INFINI_DTYPE_I64:
    case INFINI_DTYPE_U64:
        element_size = 8;
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    // 分发到对应的存储类型
    switch (element_size) {
    case 1:
        LAUNCH_BY_SIZE(uint8_t);
        break; // PackSize = 16
    case 2:
        LAUNCH_BY_SIZE(uint16_t);
        break; // PackSize = 8
    case 4:
        LAUNCH_BY_SIZE(uint32_t);
        break; // PackSize = 4 (float4)
    case 8:
        LAUNCH_BY_SIZE(uint64_t);
        break; // PackSize = 2 (double2)
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

#undef LAUNCH_BY_SIZE
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::take::nvidia
