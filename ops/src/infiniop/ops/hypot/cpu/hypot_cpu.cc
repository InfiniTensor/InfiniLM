// 引用 Hypot 专用的 CPU 头文件
#include "hypot_cpu.h"

namespace op::hypot::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto dtype = out_desc->dtype();
    if (input_desc_vec.size() != 2) {
        return INFINI_STATUS_BAD_PARAM;
    }

    const auto &input_a_desc = input_desc_vec.at(0);
    const auto &input_b_desc = input_desc_vec.at(1);
    const auto &output_shape = out_desc->shape();

    CHECK_DTYPE(dtype,
                INFINI_DTYPE_BF16, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);

    // 检查两个输入的形状是否与输出一致
    CHECK_SAME_SHAPE(output_shape, input_a_desc->shape());
    CHECK_SAME_SHAPE(output_shape, input_b_desc->shape());

    // create CPU elementwise descriptor
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
    case INFINI_DTYPE_BF16:
        return _device_info->calculate<HypotOp, bf16_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_F16:
        return _device_info->calculate<HypotOp, fp16_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_F32:
        return _device_info->calculate<HypotOp, float>(_info, output, inputs, stream);
    case INFINI_DTYPE_F64:
        return _device_info->calculate<HypotOp, double>(_info, output, inputs, stream);

    default:
        // 如果传入了整数类型或其他不支持的类型，将返回错误
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::hypot::cpu
