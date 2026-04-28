#include "index_copy_moore.h"
#include "index_copy_moore_kernel.h" // 包含 IndexCopyOp Functor 定义

#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>

#include "../../../devices/moore/moore_handle.h"

namespace op::index_copy::moore {

// ==================================================================
// 1. Kernel Wrapper Implementation
// ==================================================================

template <typename T, typename TIdx>
__global__ void index_copy_kernel(
    const size_t num_elements, // Source 元素总数 (线程任务总量)
    const size_t index_len,    // Index 长度
    const size_t inner_size,   // Stride
    const size_t dim_size,     // Output 在 dim 维度的长度
    const T *source,
    const TIdx *indices,
    T *output) {

    // idx 对应 Source 张量的线性索引
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_elements) {
        // 使用 index_copy_moore_kernel.h 中定义的 Functor
        IndexCopyOp op;
        op(idx, index_len, inner_size, dim_size, source, indices, output);
    }
}

// ==================================================================
// 2. Launcher Implementation
// ==================================================================

template <typename T, typename TIdx>
void index_copy_moore_launch(
    const IndexCopyInfo &info,
    T *output,
    const T *input,
    const T *source,
    const void *indices, // void* 传入，内部强转
    void *stream) {

    auto musa_stream = (musaStream_t)stream;
    const TIdx *indices_ptr = static_cast<const TIdx *>(indices);

    // --------------------------------------------------------------
    // 步骤 1: Copy Input -> Output
    // --------------------------------------------------------------
    // Output 初始化为 Input 的值
    size_t total_out_elements = info.outer_size() * info.dim_size() * info.inner_size();

    // 如果 input 和 output 指针不同，则执行拷贝
    if (output != input) {
        musaMemcpyAsync(output, input, total_out_elements * sizeof(T), musaMemcpyDeviceToDevice, musa_stream);
    }

    // --------------------------------------------------------------
    // 步骤 2: Scatter (Source -> Output)
    // --------------------------------------------------------------
    // 线程并行度取决于 Source 的大小
    // Source 逻辑形状: [Outer, IndexLen, Inner]
    size_t num_src_elements = info.outer_size() * info.index_len() * info.inner_size();

    if (num_src_elements == 0) {
        return;
    }

    int threads = 256;
    int blocks = (num_src_elements + threads - 1) / threads;

    index_copy_kernel<T, TIdx><<<blocks, threads, 0, musa_stream>>>(
        num_src_elements,
        info.index_len(),
        info.inner_size(),
        info.dim_size(),
        source,
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
    int64_t dim,
    infiniopTensorDescriptor_t index_desc,
    infiniopTensorDescriptor_t source_desc) {

    auto handle = reinterpret_cast<device::moore::Handle *>(handle_);

    // 使用 Info 类校验形状和类型
    auto info_result = IndexCopyInfo::create(out_desc, in_desc, dim, index_desc, source_desc);

    if (!info_result) {
        return info_result.status();
    }

    *desc_ptr = new Descriptor(
        nullptr,
        *info_result,
        0, // No workspace needed
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    const void *index,
    const void *source,
    void *stream) const {

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

// --------------------------------------------------------------
// 定义分发宏：Data Type x Index Type
// --------------------------------------------------------------
#define LAUNCH_KERNEL(T)                                    \
    do {                                                    \
        if (_info.idx_dtype() == INFINI_DTYPE_I32) {        \
            index_copy_moore_launch<T, int32_t>(            \
                _info,                                      \
                static_cast<T *>(output),                   \
                static_cast<const T *>(input),              \
                static_cast<const T *>(source),             \
                index,                                      \
                stream);                                    \
        } else if (_info.idx_dtype() == INFINI_DTYPE_I64) { \
            index_copy_moore_launch<T, int64_t>(            \
                _info,                                      \
                static_cast<T *>(output),                   \
                static_cast<const T *>(input),              \
                static_cast<const T *>(source),             \
                index,                                      \
                stream);                                    \
        } else {                                            \
            return INFINI_STATUS_BAD_TENSOR_DTYPE;          \
        }                                                   \
    } while (0)

    // --------------------------------------------------------------
    // 根据数据类型分发
    // --------------------------------------------------------------
    switch (_info.dtype()) {
    case INFINI_DTYPE_F16:
        LAUNCH_KERNEL(half);
        break;

    case INFINI_DTYPE_BF16:
        LAUNCH_KERNEL(__mt_bfloat16);
        break;

    case INFINI_DTYPE_F32:
        LAUNCH_KERNEL(float);
        break;

    case INFINI_DTYPE_F64:
        LAUNCH_KERNEL(double);
        break;

    case INFINI_DTYPE_I32:
        LAUNCH_KERNEL(int32_t);
        break;

    case INFINI_DTYPE_I64:
        LAUNCH_KERNEL(int64_t);
        break;

    case INFINI_DTYPE_I8:
        LAUNCH_KERNEL(int8_t);
        break;

    case INFINI_DTYPE_U8:
        LAUNCH_KERNEL(uint8_t);
        break;

    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

#undef LAUNCH_KERNEL

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::index_copy::moore
