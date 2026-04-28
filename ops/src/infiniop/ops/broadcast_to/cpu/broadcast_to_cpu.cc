#include "broadcast_to_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <omp.h>
#include <vector>

#include "../../../../utils/custom_types.h"

namespace op::broadcast_to::cpu {

struct Descriptor::Opaque {};

Descriptor::~Descriptor() {
    if (_opaque) {
        delete _opaque;
        _opaque = nullptr;
    }
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    const std::vector<infiniopTensorDescriptor_t> &input_descs) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto result = BroadcastToInfo::create(out_desc, input_descs);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        new Opaque(),
        result.take(),
        0, // CPU 实现不需要 workspace
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

template <typename T>
void calculate_cpu_impl(
    const BroadcastToInfo &info,
    void *output,
    const void *input) {

    size_t count = info.count();
    int ndim = info.ndim();

    auto out_ptr = reinterpret_cast<T *>(output);
    auto in_ptr = reinterpret_cast<const T *>(input);

// 并行遍历输出的每一个元素
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < (ptrdiff_t)count; ++i) {
        size_t temp_idx = i;
        size_t input_offset = 0;

        // 坐标变换：Output Linear Index -> Coordinate -> Input Linear Offset
        for (int d = 0; d < ndim; ++d) {
            size_t out_stride = info._out_strides[d];
            size_t coord = temp_idx / out_stride;
            temp_idx %= out_stride;
            input_offset += coord * info._in_strides[d];
        }

        // 3. 赋值
        out_ptr[i] = in_ptr[input_offset];
    }
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

    switch (dtype) {
    case INFINI_DTYPE_F32:
        cpu::calculate_cpu_impl<float>(_info, output, input);
        break;
    case INFINI_DTYPE_F64:
        cpu::calculate_cpu_impl<double>(_info, output, input);
        break;
    case INFINI_DTYPE_F16:
        cpu::calculate_cpu_impl<fp16_t>(_info, output, input);
        break;
    case INFINI_DTYPE_BF16:
        cpu::calculate_cpu_impl<bf16_t>(_info, output, input);
        break;
    case INFINI_DTYPE_I64:
        cpu::calculate_cpu_impl<int64_t>(_info, output, input);
        break;
    case INFINI_DTYPE_I32:
        cpu::calculate_cpu_impl<int32_t>(_info, output, input);
        break;
    case INFINI_DTYPE_U8:
        cpu::calculate_cpu_impl<uint8_t>(_info, output, input);
        break;
    case INFINI_DTYPE_I8:
        cpu::calculate_cpu_impl<int8_t>(_info, output, input);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::broadcast_to::cpu
