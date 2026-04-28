#include "equal_moore.h"

#include "../../../elementwise/moore/elementwise_moore.h"

#include "equal_moore_kernel.h"

namespace op::equal::moore {
namespace {

inline bool can_use_contiguous_fast_path(const op::elementwise::ElementwiseInfo &info) {
    if (!info.isOutputContiguous()) {
        return false;
    }
    const bool *input_contiguous = info.getInputContiguous();
    const bool *input_broadcasted = info.getInputBroadcasted();
    for (size_t i = 0; i < 2; ++i) {
        if (!input_contiguous[i] || input_broadcasted[i]) {
            return false;
        }
    }
    return true;
}

template <typename Tout, typename Tin>
INFINIOP_MOORE_KERNEL equal_contiguous_kernel(size_t numel, Tout *output, const Tin *a, const Tin *b) {
    const auto op = op::equal::moore::EqualOp{};
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (; idx < numel; idx += stride) {
        output[idx] = op.template operator()<Tout, Tin>(a[idx], b[idx]);
    }
}

template <typename Tout, typename Tin>
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
    equal_contiguous_kernel<Tout, Tin><<<grid, kBlockSize, 0, musa_stream>>>(
        numel,
        reinterpret_cast<Tout *>(output),
        reinterpret_cast<const Tin *>(inputs[0]),
        reinterpret_cast<const Tin *>(inputs[1]));
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

    const auto &a_desc = input_desc_vec.at(0);
    auto compute_dtype = a_desc->dtype();
    auto out_dtype = out_desc->dtype();

    const auto &b_desc = input_desc_vec.at(1);
    const auto &c_shape = out_desc->shape();
    const auto &a_shape = a_desc->shape();
    const auto &b_shape = b_desc->shape();

    CHECK_DTYPE(compute_dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16,
                INFINI_DTYPE_I32, INFINI_DTYPE_I64, INFINI_DTYPE_F64);

    CHECK_DTYPE(out_dtype, INFINI_DTYPE_BOOL);

    CHECK_SAME_SHAPE(c_shape, a_shape, b_shape);

    // create MOORE elementwise descriptor
    CREATE_ELEMENTWISE_MOORE_DESCRIPTOR(handle, compute_dtype, out_desc, input_desc_vec)

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {
    if (can_use_contiguous_fast_path(_info)) {
        size_t numel = _info.getOutputSize();
        switch (_dtype) {
        case INFINI_DTYPE_F16:
            return launch_fast_path<bool, half>(numel, output, inputs, stream);
        case INFINI_DTYPE_BF16:
            return launch_fast_path<bool, cuda_bfloat16>(numel, output, inputs, stream);
        case INFINI_DTYPE_F32:
            return launch_fast_path<bool, float>(numel, output, inputs, stream);
        case INFINI_DTYPE_I32:
            return launch_fast_path<bool, int32_t>(numel, output, inputs, stream);
        case INFINI_DTYPE_I64:
            return launch_fast_path<bool, int64_t>(numel, output, inputs, stream);
        case INFINI_DTYPE_F64:
            return launch_fast_path<bool, double>(numel, output, inputs, stream);
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    }

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        return _device_info->calculate<256, moore::EqualOp, bool, half, half>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_BF16:
        return _device_info->calculate<256, moore::EqualOp, bool, cuda_bfloat16, cuda_bfloat16>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_F32:
        return _device_info->calculate<256, moore::EqualOp, bool, float, float>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_I32:
        return _device_info->calculate<256, moore::EqualOp, bool, int32_t, int32_t>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_I64:
        return _device_info->calculate<256, moore::EqualOp, bool, int64_t, int64_t>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_F64:
        return _device_info->calculate<256, moore::EqualOp, bool, double, double>(_info, workspace, output, inputs, stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::equal::moore
