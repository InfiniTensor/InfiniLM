#include "../../../elementwise/nvidia/elementwise_nvidia.cuh"

#include "../cuda/kernel.cuh"
#include "hypot_nvidia.cuh"

namespace op::hypot::nvidia {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) {

    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto dtype = out_desc->dtype();

    if (input_desc_vec.size() != 2) {
        return INFINI_STATUS_BAD_PARAM;
    }

    const auto &input_a_desc = input_desc_vec.at(0);
    const auto &input_b_desc = input_desc_vec.at(1);
    const auto &output_shape = out_desc->shape();

    CHECK_DTYPE(dtype,
                INFINI_DTYPE_BF16, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);

    CHECK_SAME_SHAPE(output_shape, input_a_desc->shape());
    CHECK_SAME_SHAPE(output_shape, input_b_desc->shape());

    // 创建描述符
    CREATE_ELEMENTWISE_CUDA_DESCRIPTOR(handle, dtype, out_desc, input_desc_vec)

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

    // -----------------------------------------------------------
    // 算子分发：将 FloorOp 替换为 HypotOp
    // -----------------------------------------------------------
    switch (_dtype) {
    // === 浮点类型 ===
    case INFINI_DTYPE_BF16:
        // 注意：cuda::HypotOp 对应我们在 hypot_cuda.h 中定义的 Functor
        return _device_info->calculate<256, cuda::HypotOp, cuda_bfloat16>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_F16:
        return _device_info->calculate<256, cuda::HypotOp, half>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_F32:
        return _device_info->calculate<256, cuda::HypotOp, float>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_F64:
        return _device_info->calculate<256, cuda::HypotOp, double>(_info, workspace, output, inputs, stream);

        // 【修改点 4】移除了整数类型的 Case

    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::hypot::nvidia
