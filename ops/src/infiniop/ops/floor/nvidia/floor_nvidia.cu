#include "../../../elementwise/nvidia/elementwise_nvidia.cuh"

// 引入核心计算 Functor (我们在 src/infiniop/ops/floor/cuda/floor_cuda.h 中定义的)
#include "../cuda/kernel.cuh"
#include "floor_nvidia.cuh"

namespace op::floor::nvidia {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) {

    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto dtype = out_desc->dtype();

    const auto &input_desc = input_desc_vec.at(0);
    const auto &output_shape = out_desc->shape();
    const auto &input_shape = input_desc->shape();
    CHECK_DTYPE(dtype,
                INFINI_DTYPE_BF16, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64,
                INFINI_DTYPE_I8, INFINI_DTYPE_U8,
                INFINI_DTYPE_I16, INFINI_DTYPE_U16,
                INFINI_DTYPE_I32, INFINI_DTYPE_U32,
                INFINI_DTYPE_I64, INFINI_DTYPE_U64);

    CHECK_SAME_SHAPE(output_shape, input_shape);

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
    // 2. 算子分发：将 GeluOp 替换为 FloorOp
    //    模板参数 <256, ...> 表示 CUDA Block Size
    // -----------------------------------------------------------
    switch (_dtype) {
    // === 浮点类型 ===
    case INFINI_DTYPE_BF16:
        return _device_info->calculate<256, cuda::FloorOp, cuda_bfloat16>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_F16:
        return _device_info->calculate<256, cuda::FloorOp, half>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_F32:
        return _device_info->calculate<256, cuda::FloorOp, float>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_F64:
        return _device_info->calculate<256, cuda::FloorOp, double>(_info, workspace, output, inputs, stream);

    // === 整数类型 (调用 FloorOp 也会正确处理，直接返回原值) ===
    case INFINI_DTYPE_I8:
        return _device_info->calculate<256, cuda::FloorOp, int8_t>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_U8:
        return _device_info->calculate<256, cuda::FloorOp, uint8_t>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_I16:
        return _device_info->calculate<256, cuda::FloorOp, int16_t>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_U16:
        return _device_info->calculate<256, cuda::FloorOp, uint16_t>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_I32:
        return _device_info->calculate<256, cuda::FloorOp, int32_t>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_U32:
        return _device_info->calculate<256, cuda::FloorOp, uint32_t>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_I64:
        return _device_info->calculate<256, cuda::FloorOp, int64_t>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_U64:
        return _device_info->calculate<256, cuda::FloorOp, uint64_t>(_info, workspace, output, inputs, stream);

    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::floor::nvidia
