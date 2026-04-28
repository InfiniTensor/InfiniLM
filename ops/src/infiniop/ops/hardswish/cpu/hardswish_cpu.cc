#include "hardswish_cpu.h"

#include <cstddef>

namespace op::hardswish::cpu {
namespace {

inline bool can_use_contiguous_fast_path(const op::elementwise::ElementwiseInfo &info) {
    return info.isOutputContiguous() && info.getInputSize() == 1 && info.getInputContiguous()[0] && !info.getInputBroadcasted()[0];
}

template <typename T>
infiniStatus_t launch_contiguous_cpu(const op::elementwise::ElementwiseInfo &info,
                                     void *output,
                                     const std::vector<const void *> &inputs) {
    const T *in = reinterpret_cast<const T *>(inputs[0]);
    T *out = reinterpret_cast<T *>(output);
    const ptrdiff_t size = static_cast<ptrdiff_t>(info.getOutputSize());

#pragma omp parallel for if (size > 1024)
    for (ptrdiff_t i = 0; i < size; ++i) {
        out[i] = HardSwishOp{}(in[i]);
    }
    return INFINI_STATUS_SUCCESS;
}

} // namespace

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

    CHECK_DTYPE(dtype, INFINI_DTYPE_BF16, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);

    CHECK_SAME_SHAPE(output_shape, input_shape);

    CREATE_ELEMENTWISE_CPU_DESCRIPTOR(handle, dtype, out_desc, input_desc_vec);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {

    const bool fast_path = can_use_contiguous_fast_path(_info);
    if (fast_path) {
        switch (_dtype) {
        case INFINI_DTYPE_BF16:
            return launch_contiguous_cpu<bf16_t>(_info, output, inputs);
        case INFINI_DTYPE_F16:
            return launch_contiguous_cpu<fp16_t>(_info, output, inputs);
        case INFINI_DTYPE_F32:
            return launch_contiguous_cpu<float>(_info, output, inputs);
        case INFINI_DTYPE_F64:
            return launch_contiguous_cpu<double>(_info, output, inputs);
        default:
            break;
        }
    }

    switch (_dtype) {
    case INFINI_DTYPE_BF16:
        return _device_info->calculate<HardSwishOp, bf16_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_F16:
        return _device_info->calculate<HardSwishOp, fp16_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_F32:
        return _device_info->calculate<HardSwishOp, float>(_info, output, inputs, stream);
    case INFINI_DTYPE_F64:
        return _device_info->calculate<HardSwishOp, double>(_info, output, inputs, stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::hardswish::cpu
