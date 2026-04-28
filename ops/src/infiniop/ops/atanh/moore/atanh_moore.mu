#include "../../../elementwise/moore/elementwise_moore.h"

#include "atanh_moore.h"
#include "atanh_moore_kernel.h"

namespace op::atanh::moore {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) {

    // 1. 转换 Handle 为 Moore 平台类型
    auto handle = reinterpret_cast<device::moore::Handle *>(handle_);
    auto dtype = out_desc->dtype();

    const auto &a_desc = input_desc_vec.at(0);
    const auto &y_shape = out_desc->shape();
    const auto &a_shape = a_desc->shape();

    // 2. 检查数据类型支持情况
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16, INFINI_DTYPE_F64);

    // 3. 校验 Shape 一致性
    CHECK_SAME_SHAPE(y_shape, a_shape);

    // 4. 创建 Moore 平台的 Elementwise 描述符
    CREATE_ELEMENTWISE_MOORE_DESCRIPTOR(handle, dtype, out_desc, input_desc_vec)

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    // 5. 根据数据类型分发到具体的 MUSA Kernel 逻辑
    switch (_dtype) {
    case INFINI_DTYPE_F16:
        return _device_info->calculate<256, moore::AtanhOp, half>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_BF16:
        // 注意：这里将 nv_bfloat16 替换为 Moore 环境下的 bfloat16 类型名
        return _device_info->calculate<256, moore::AtanhOp, cuda_bfloat16>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_F32:
        return _device_info->calculate<256, moore::AtanhOp, float>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_F64:
        return _device_info->calculate<256, moore::AtanhOp, double>(_info, workspace, output, inputs, stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::atanh::moore
