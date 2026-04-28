#include <cstdint>
#include <type_traits>

#include "equal_cpu.h"

namespace op::equal::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

    const auto &a_desc = input_desc_vec.at(0);
    const auto &b_desc = input_desc_vec.at(1);
    auto compute_dtype = a_desc->dtype();
    auto out_dtype = out_desc->dtype();

    if (compute_dtype != b_desc->dtype()) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    CHECK_DTYPE(out_dtype, INFINI_DTYPE_BOOL);

    CHECK_DTYPE(compute_dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64,
                INFINI_DTYPE_BF16, INFINI_DTYPE_I32, INFINI_DTYPE_I64);

    const auto &c_shape = out_desc->shape();
    const auto &a_shape = a_desc->shape();
    const auto &b_shape = b_desc->shape();

    CHECK_SAME_SHAPE(c_shape, a_shape, b_shape);

    CREATE_ELEMENTWISE_CPU_DESCRIPTOR(handle, compute_dtype, out_desc, input_desc_vec);

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
        return _device_info->calculate<EqualOp, bool, fp16_t, fp16_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_F32:
        return _device_info->calculate<EqualOp, bool, float, float>(_info, output, inputs, stream);
    case INFINI_DTYPE_F64:
        return _device_info->calculate<EqualOp, bool, double, double>(_info, output, inputs, stream);
    case INFINI_DTYPE_BF16:
        return _device_info->calculate<EqualOp, bool, bf16_t, bf16_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_I32:
        return _device_info->calculate<EqualOp, bool, int32_t, int32_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_I64:
        return _device_info->calculate<EqualOp, bool, int64_t, int64_t>(_info, output, inputs, stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::equal::cpu
