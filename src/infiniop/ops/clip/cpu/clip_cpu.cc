#include "clip_cpu.h"

namespace op::clip::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto dtype = out_desc->dtype();

    const auto &in_desc = input_desc_vec.at(0);
    const auto &min_desc = input_desc_vec.at(1);
    const auto &max_desc = input_desc_vec.at(2);
    const auto &out_shape = out_desc->shape();
    const auto &in_shape = in_desc->shape();
    const auto &min_shape = min_desc->shape();
    const auto &max_shape = max_desc->shape();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);
    CHECK_SAME_SHAPE(out_shape, in_shape);
    CHECK_SAME_SHAPE(out_shape, min_shape);
    CHECK_SAME_SHAPE(out_shape, max_shape);

    CREATE_ELEMENTWISE_CPU_DESCRIPTOR(handle, dtype, out_desc, input_desc_vec);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        return _device_info->calculate<ClipOp, fp16_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_F32:
        return _device_info->calculate<ClipOp, float>(_info, output, inputs, stream);
    case INFINI_DTYPE_F64:
        return _device_info->calculate<ClipOp, double>(_info, output, inputs, stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::clip::cpu
