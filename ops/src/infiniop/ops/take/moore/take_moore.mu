#include "take_moore.h"
#include "take_moore_kernel.h" // 包含 TakeOp 结构体定义

#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>

#include "../../../devices/moore/moore_handle.h"

namespace op::take::moore {

// ==================================================================
// 1. Kernel Wrapper Implementation
// ==================================================================

// 这是一个 Global Kernel 包装器，它调用 take_moore_kernel.h 中的 TakeOp Functor
template <typename T, typename TIdx>
__global__ void take_kernel(
    const size_t num_out, // 输出元素总数
    const size_t num_in,  // 输入元素总数 (用于边界检查)
    const T *input,       // 输入数据
    const TIdx *indices,  // 索引数据
    T *output) {          // 输出数据

    // idx 对应输出张量和索引张量的线性索引
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_out) {
        // 使用 take_moore_kernel.h 中定义的 Functor
        TakeOp op;
        op(idx, num_in, input, indices, output);
    }
}

// ==================================================================
// 2. Launcher Implementation
// ==================================================================

template <typename T, typename TIdx>
void take_moore_launch(
    const TakeInfo &info,
    T *output,
    const T *input,
    const void *indices, // indices 传入时是 void*，需要强转为 TIdx*
    void *stream) {

    size_t num_out = info.num_out();
    size_t num_in = info.num_in();

    // 强转索引指针
    const TIdx *indices_ptr = static_cast<const TIdx *>(indices);

    int threads = 256;
    // 使用 size_t 防止溢出，但 GridDim 必须是 int
    // 如果 num_out 非常大，可能需要 stride loop，此处假设在 Grid 限制内
    int blocks = (num_out + threads - 1) / threads;

    take_kernel<T, TIdx><<<blocks, threads, 0, (musaStream_t)stream>>>(
        num_out,
        num_in,
        input,
        indices_ptr,
        output);
}

// ==================================================================
// 3. Descriptor Implementation
// ==================================================================

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t in_desc,
    infiniopTensorDescriptor_t indices_desc) {

    auto handle = reinterpret_cast<device::moore::Handle *>(handle_);

    // 使用 TakeInfo::create 校验形状和类型
    auto info_result = TakeInfo::create(out_desc, in_desc, indices_desc);

    if (!info_result) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
    *desc_ptr = new Descriptor(
        nullptr,
        *info_result, // 解包 Result 获取 Info 对象
        0,            // Take 算子不需要 Workspace
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    const void *indices,
    void *stream) const {

    // 1. 基础检查
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

// 2. 定义分发宏：先确定 Data Type，再确定 Index Type
// T 是数据类型 (float, half, etc.)
#define LAUNCH_TAKE_KERNEL(T)                                            \
    do {                                                                 \
        if (_info.idx_dtype() == INFINI_DTYPE_I32) {                     \
            take_moore_launch<T, int32_t>(_info,                         \
                                          static_cast<T *>(output),      \
                                          static_cast<const T *>(input), \
                                          indices, stream);              \
        } else if (_info.idx_dtype() == INFINI_DTYPE_I64) {              \
            take_moore_launch<T, int64_t>(_info,                         \
                                          static_cast<T *>(output),      \
                                          static_cast<const T *>(input), \
                                          indices, stream);              \
        } else {                                                         \
            return INFINI_STATUS_BAD_TENSOR_DTYPE;                       \
        }                                                                \
    } while (0)

    // 3. 根据数据类型分发
    switch (_info.dtype()) {
    case INFINI_DTYPE_F16:
        LAUNCH_TAKE_KERNEL(half);
        break;

    case INFINI_DTYPE_BF16:
        LAUNCH_TAKE_KERNEL(__mt_bfloat16);
        break;

    case INFINI_DTYPE_F32:
        LAUNCH_TAKE_KERNEL(float);
        break;

    case INFINI_DTYPE_F64:
        LAUNCH_TAKE_KERNEL(double);
        break;

    case INFINI_DTYPE_I32:
        LAUNCH_TAKE_KERNEL(int32_t);
        break;

    case INFINI_DTYPE_I64:
        LAUNCH_TAKE_KERNEL(int64_t);
        break;

    case INFINI_DTYPE_I8:
        LAUNCH_TAKE_KERNEL(int8_t);
        break;

    case INFINI_DTYPE_U8:
        LAUNCH_TAKE_KERNEL(uint8_t);
        break;

    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

#undef LAUNCH_TAKE_KERNEL

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::take::moore
