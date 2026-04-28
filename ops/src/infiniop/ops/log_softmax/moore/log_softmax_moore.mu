#include "../../../devices/moore/moore_handle.h"
#include "log_softmax_moore.h"
#include "log_softmax_moore_kernel.h"
#include <algorithm>
#include <cstdint>
#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
namespace op::log_softmax::moore {
template <typename T>
void launch_kernel(
    void *output,
    const void *input,
    const LogSoftmaxInfo &info,
    void *stream) {

    // 1. 准备指针
    auto in_ptr = reinterpret_cast<const T *>(input);
    auto out_ptr = reinterpret_cast<T *>(output);

    // MUSA 流类型转换
    auto musa_stream = reinterpret_cast<musaStream_t>(stream);

    // 2. 准备形状参数
    size_t dim_size = info.dim_size();
    size_t outer_size = info.outer_size();
    size_t inner_size = info.inner_size();
    size_t total_slices = outer_size * inner_size;
    unsigned int threads_per_block = 256;

    // 如果 dim_size 很小，可以适当减小 block size，但不要小于 32 (Warp Size)
    if (dim_size < 256) {
        threads_per_block = 128;
    }
    if (dim_size < 128) {
        threads_per_block = 64;
    }
    if (dim_size < 64) {
        threads_per_block = 32;
    }
    op::log_softmax::moore::log_softmax_kernel<T>
        <<<total_slices, threads_per_block, 0, musa_stream>>>(
            out_ptr,
            in_ptr,
            dim_size,
            inner_size);
}
struct Descriptor::Opaque {};

Descriptor::~Descriptor() {
    if (_opaque) {
        delete _opaque;
    }
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle, Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    int dim) {

    auto info_result = LogSoftmaxInfo::create(output_desc, input_desc, dim);
    if (!info_result) {
        return info_result.status();
    }

    // LogSoftmax 此实现为 Online 算法，不需要额外的 Workspace
    size_t workspace_size = 0;

    *desc_ptr = new Descriptor(new Opaque(), info_result.take(), workspace_size, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream) const {

    auto dtype = _info.dtype();

    switch (dtype) {
    case INFINI_DTYPE_F16:
        // MUSA 使用 half
        launch_kernel<half>(output, input, _info, stream);
        break;
    case INFINI_DTYPE_BF16:
        launch_kernel<__mt_bfloat16>(output, input, _info, stream);
        break;
    case INFINI_DTYPE_F32:
        launch_kernel<float>(output, input, _info, stream);
        break;
    case INFINI_DTYPE_F64:
        launch_kernel<double>(output, input, _info, stream);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::log_softmax::moore
