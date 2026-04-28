#include "addcmul_cpu.h"

namespace op::addcmul::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec,
    float value) { // 额外接收 value 参数

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto dtype = out_desc->dtype();

    // 1. 类型检查
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16);

    // 2. 形状检查 (仿照 atanh，这里至少检查第一个输入)
    const auto &y_shape = out_desc->shape();
    for (const auto &in_desc : input_desc_vec) {
        CHECK_SAME_SHAPE(y_shape, in_desc->shape());
    }

    // 3. 使用通用的 Elementwise 宏创建描述符
    // 该宏会实例化 Descriptor 并将其赋值给 *desc_ptr
    CREATE_ELEMENTWISE_CPU_DESCRIPTOR(handle, dtype, out_desc, input_desc_vec);

    // 4. 将标量属性 value 存入 Descriptor 内部
    (*desc_ptr)->_value = value;

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {

    // 仿照 atanh，使用 switch 分发不同数据类型
    // 这里的模板参数是 AddcmulOp，它在 addcmul_cpu.h 中定义
    switch (_dtype) {
    case INFINI_DTYPE_F16:
        return _device_info->calculate<AddcmulOp, fp16_t>(_info, output, inputs, stream, _value);
    case INFINI_DTYPE_F32:
        return _device_info->calculate<AddcmulOp, float>(_info, output, inputs, stream, _value);
    case INFINI_DTYPE_F64:
        return _device_info->calculate<AddcmulOp, double>(_info, output, inputs, stream, _value);
    case INFINI_DTYPE_BF16:
        return _device_info->calculate<AddcmulOp, bf16_t>(_info, output, inputs, stream, _value);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::addcmul::cpu
