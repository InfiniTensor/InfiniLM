#include "../../../elementwise/moore/elementwise_moore.h"

#include "reciprocal_moore.h"
#include "reciprocal_moore_kernel.h"

namespace op::reciprocal::moore {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) {

    // 1. 解析 Moore (MUSA) 句柄
    auto handle = reinterpret_cast<device::moore::Handle *>(handle_);
    auto dtype = out_desc->dtype();

    const auto &y_desc = out_desc;
    const auto &x_desc = input_desc_vec.at(0);
    const auto &y_shape = y_desc->shape();
    const auto &x_shape = x_desc->shape();

    // 2. 校验数据类型：Moore 平台同样在浮点类型上执行倒数运算
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16, INFINI_DTYPE_F64);

    CHECK_SAME_SHAPE(y_shape, x_shape);

    // 3. 使用 Moore 平台的 Elementwise 描述符创建宏
    // 该宏会自动处理 MUSA 后端的算子元数据初始化
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

    // 4. 分发至 Moore 特化的计算逻辑
    // 注意：cuda::ReciprocalOp 替换为 moore::ReciprocalOp
    switch (_dtype) {
    case INFINI_DTYPE_F16:
        return _device_info->calculate<256, moore::ReciprocalOp, half>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_BF16:
        // 确保使用 Moore 环境下的 bfloat16 类型定义
        return _device_info->calculate<256, moore::ReciprocalOp, cuda_bfloat16>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_F32:
        return _device_info->calculate<256, moore::ReciprocalOp, float>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_F64:
        return _device_info->calculate<256, moore::ReciprocalOp, double>(_info, workspace, output, inputs, stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::reciprocal::moore
