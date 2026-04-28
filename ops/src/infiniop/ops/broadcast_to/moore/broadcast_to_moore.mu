#include "../../../devices/moore/moore_handle.h"
#include "broadcast_to_moore.h"
#include "broadcast_to_moore_kernel.h"
#include <algorithm>
#include <cstdint>
#include <vector>

namespace op::broadcast_to::moore {

// ==================================================================
// Kernel Launch Logic
// ==================================================================
template <typename T>
void launch_kernel(
    void *output,
    const void *input,
    const BroadcastToInfo &info,
    void *stream) {

    auto in_ptr = reinterpret_cast<const T *>(input);
    auto out_ptr = reinterpret_cast<T *>(output);

    auto musa_stream = reinterpret_cast<musaStream_t>(stream);

    // 复制 strides 到 Kernel 定义的结构体中
    op::broadcast_to::moore::BroadcastStrides strides;
    for (int i = 0; i < BroadcastToInfo::MAX_DIM; ++i) {
        strides.out_strides[i] = info._out_strides[i];
        strides.in_strides[i] = info._in_strides[i];
    }

    size_t count = info.count();
    size_t block_size = 256;
    size_t grid_size = (count + block_size - 1) / block_size;

    op::broadcast_to::moore::broadcast_kernel<T>
        <<<grid_size, block_size, 0, musa_stream>>>(
            out_ptr,
            in_ptr,
            info.ndim(),
            count,
            strides);
}

struct Descriptor::Opaque {};

Descriptor::~Descriptor() {
    if (_opaque) {
        delete _opaque;
    }
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    const std::vector<infiniopTensorDescriptor_t> &input_descs) {

    auto handle = reinterpret_cast<device::moore::Handle *>(handle_);

    auto info_result = BroadcastToInfo::create(out_desc, input_descs);
    if (!info_result) {
        return info_result.status();
    }
    size_t workspace_size = 0;

    *desc_ptr = new Descriptor(
        new Opaque(),
        info_result.take(),
        workspace_size,
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const std::vector<const void *> &inputs,
    void *stream) const {

    if (inputs.size() != 1) {
        return INFINI_STATUS_BAD_PARAM;
    }
    const void *input = inputs[0];

    auto dtype = _info.dtype();

    // 3. 根据数据类型分发 Kernel
    switch (dtype) {
    case INFINI_DTYPE_F16:
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
    case INFINI_DTYPE_I64:
        launch_kernel<int64_t>(output, input, _info, stream);
        break;
    case INFINI_DTYPE_I32:
        launch_kernel<int32_t>(output, input, _info, stream);
        break;
    case INFINI_DTYPE_U8:
        launch_kernel<uint8_t>(output, input, _info, stream);
        break;
    case INFINI_DTYPE_I8:
        launch_kernel<int8_t>(output, input, _info, stream);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::broadcast_to::moore
