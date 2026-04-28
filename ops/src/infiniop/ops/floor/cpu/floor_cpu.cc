// 【修改点 1】引用 Floor 专用的 CPU 头文件
#include "floor_cpu.h"

// 【修改点 2】命名空间必须是 floor，否则 operator.cc 找不到定义
namespace op::floor::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto dtype = out_desc->dtype();

    const auto &input_desc = input_desc_vec.at(0);
    const auto &output_shape = out_desc->shape();
    const auto &input_shape = input_desc->shape();

    // 【修改点 3】Floor 算子通常支持浮点和整数
    // (整数做 floor 结果不变，但为了通用性建议加上)
    CHECK_DTYPE(dtype,
                INFINI_DTYPE_BF16, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64,
                INFINI_DTYPE_I8, INFINI_DTYPE_U8,
                INFINI_DTYPE_I16, INFINI_DTYPE_U16,
                INFINI_DTYPE_I32, INFINI_DTYPE_U32,
                INFINI_DTYPE_I64, INFINI_DTYPE_U64);

    CHECK_SAME_SHAPE(output_shape, input_shape);

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

    // 【修改点 4】分发计算：将 GeluOp 替换为 FloorOp
    switch (_dtype) {
    // === 浮点类型 ===
    case INFINI_DTYPE_BF16:
        return _device_info->calculate<FloorOp, bf16_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_F16:
        return _device_info->calculate<FloorOp, fp16_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_F32:
        return _device_info->calculate<FloorOp, float>(_info, output, inputs, stream);
    case INFINI_DTYPE_F64:
        return _device_info->calculate<FloorOp, double>(_info, output, inputs, stream);

    // === 整数类型 (直接调用 FloorOp，因为 FloorOp 对整数是恒等映射) ===
    case INFINI_DTYPE_I8:
        return _device_info->calculate<FloorOp, int8_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_U8:
        return _device_info->calculate<FloorOp, uint8_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_I16:
        return _device_info->calculate<FloorOp, int16_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_U16:
        return _device_info->calculate<FloorOp, uint16_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_I32:
        return _device_info->calculate<FloorOp, int32_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_U32:
        return _device_info->calculate<FloorOp, uint32_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_I64:
        return _device_info->calculate<FloorOp, int64_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_U64:
        return _device_info->calculate<FloorOp, uint64_t>(_info, output, inputs, stream);

    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::floor::cpu
