#include "hardswish_moore.h"

#include "../../../elementwise/moore/elementwise_moore.h"

#include "hardswish_moore_kernel.h"

namespace op::hardswish::moore {
namespace {

inline bool can_use_contiguous_fast_path(const op::elementwise::ElementwiseInfo &info) {
    return info.isOutputContiguous() && info.getInputSize() == 1 &&
           info.getInputContiguous()[0] && !info.getInputBroadcasted()[0];
}

template <typename T>
INFINIOP_MOORE_KERNEL hardswish_contiguous_kernel(size_t numel, T *out, const T *in) {
    const auto op = op::hardswish::moore::HardSwishOp{};
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (; idx < numel; idx += stride) {
        out[idx] = op(in[idx]);
    }
}

template <typename T>
infiniStatus_t launch_fast_path(size_t numel,
                                void *output,
                                const std::vector<const void *> &inputs,
                                void *stream) {
    if (numel == 0) {
        return INFINI_STATUS_SUCCESS;
    }

    constexpr int kBlockSize = 256;
    int grid = static_cast<int>((numel + kBlockSize - 1) / kBlockSize);
    if (grid > 65535) {
        grid = 65535;
    }

    auto musa_stream = reinterpret_cast<musaStream_t>(stream);
    hardswish_contiguous_kernel<T><<<grid, kBlockSize, 0, musa_stream>>>(
        numel,
        reinterpret_cast<T *>(output),
        reinterpret_cast<const T *>(inputs[0]));
    return INFINI_STATUS_SUCCESS;
}

} // namespace

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) {

    auto handle = reinterpret_cast<device::moore::Handle *>(handle_);
    auto dtype = out_desc->dtype();

    const auto &input_desc = input_desc_vec.at(0);
    const auto &output_shape = out_desc->shape();
    const auto &input_shape = input_desc->shape();

    CHECK_DTYPE(dtype, INFINI_DTYPE_BF16, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);

    CHECK_SAME_SHAPE(output_shape, input_shape);

    // create MOORE elementwise descriptor
    CREATE_ELEMENTWISE_MOORE_DESCRIPTOR(handle, dtype, out_desc, input_desc_vec)

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
            return launch_fast_path<cuda_bfloat16>(_info.getOutputSize(), output, inputs, stream);
        case INFINI_DTYPE_F16:
            return launch_fast_path<half>(_info.getOutputSize(), output, inputs, stream);
        case INFINI_DTYPE_F32:
            return launch_fast_path<float>(_info.getOutputSize(), output, inputs, stream);
        case INFINI_DTYPE_F64:
            return launch_fast_path<double>(_info.getOutputSize(), output, inputs, stream);
        default:
            break;
        }
    }

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    switch (_dtype) {
    case INFINI_DTYPE_BF16:
        return _device_info->calculate<256, moore::HardSwishOp, cuda_bfloat16>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_F16:
        return _device_info->calculate<256, moore::HardSwishOp, half>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_F32:
        return _device_info->calculate<256, moore::HardSwishOp, float>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_F64:
        return _device_info->calculate<256, moore::HardSwishOp, double>(_info, workspace, output, inputs, stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::hardswish::moore
