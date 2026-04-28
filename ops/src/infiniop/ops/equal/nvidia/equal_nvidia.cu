#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "../../../elementwise/nvidia/elementwise_nvidia.cuh"

#include "../cuda/kernel.cuh"
#include "equal_nvidia.cuh"

namespace {

template <typename Tout, typename Tin>
INFINIOP_CUDA_KERNEL FastEqualKernel(size_t n, Tout *output, const Tin *a, const Tin *b) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    op::equal::cuda::EqualOp op{};
    for (; idx < n; idx += stride) {
        output[idx] = op.template operator()<Tout, Tin>(a[idx], b[idx]);
    }
}

template <typename Tout, typename Tin>
infiniStatus_t launchFastEqualKernel(size_t numel,
                                     void *output,
                                     const std::vector<const void *> &inputs,
                                     void *stream) {
    if (numel == 0) {
        return INFINI_STATUS_SUCCESS;
    }
    constexpr int block = 256;
    int grid = static_cast<int>((numel + block - 1) / block);
    grid = std::min(grid, 65535);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    FastEqualKernel<Tout, Tin><<<grid, block, 0, cuda_stream>>>(
        numel,
        reinterpret_cast<Tout *>(output),
        reinterpret_cast<const Tin *>(inputs[0]),
        reinterpret_cast<const Tin *>(inputs[1]));
    auto err = cudaGetLastError();
    return err == cudaSuccess ? INFINI_STATUS_SUCCESS : INFINI_STATUS_INTERNAL_ERROR;
}

} // namespace

namespace op::equal::nvidia {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) {

    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);

    const auto &a_desc = input_desc_vec.at(0);
    auto compute_dtype = a_desc->dtype();
    auto out_dtype = out_desc->dtype();

    const auto &b_desc = input_desc_vec.at(1);
    const auto &c_shape = out_desc->shape();
    const auto &a_shape = a_desc->shape();
    const auto &b_shape = b_desc->shape();

    CHECK_DTYPE(compute_dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16,
                INFINI_DTYPE_I32, INFINI_DTYPE_I64, INFINI_DTYPE_F64);

    CHECK_DTYPE(out_dtype, INFINI_DTYPE_BOOL, INFINI_DTYPE_U8, INFINI_DTYPE_I8);

    CHECK_SAME_SHAPE(c_shape, a_shape, b_shape);

    CREATE_ELEMENTWISE_CUDA_DESCRIPTOR(handle, compute_dtype, out_desc, input_desc_vec)

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {

    bool fast_path = _info.isOutputContiguous();
    if (fast_path) {
        const bool *input_contiguous = _info.getInputContiguous();
        const bool *input_broadcasted = _info.getInputBroadcasted();
        for (size_t i = 0; i < 2; ++i) {
            fast_path &= input_contiguous[i] && !input_broadcasted[i];
        }
    }

    if (fast_path) {
        size_t numel = _info.getOutputSize();
        switch (_dtype) {
        case INFINI_DTYPE_F16:
            return launchFastEqualKernel<bool, half>(numel, output, inputs, stream);
        case INFINI_DTYPE_BF16:
            return launchFastEqualKernel<bool, cuda_bfloat16>(numel, output, inputs, stream);
        case INFINI_DTYPE_F32:
            return launchFastEqualKernel<bool, float>(numel, output, inputs, stream);
        case INFINI_DTYPE_I32:
            return launchFastEqualKernel<bool, int32_t>(numel, output, inputs, stream);
        case INFINI_DTYPE_I64:
            return launchFastEqualKernel<bool, int64_t>(numel, output, inputs, stream);
        case INFINI_DTYPE_F64:
            return launchFastEqualKernel<bool, double>(numel, output, inputs, stream);
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    }

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        return _device_info->calculate<256, cuda::EqualOp, bool, half, half>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_BF16:
        return _device_info->calculate<256, cuda::EqualOp, bool, cuda_bfloat16, cuda_bfloat16>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_F32:
        return _device_info->calculate<256, cuda::EqualOp, bool, float, float>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_I32:
        return _device_info->calculate<256, cuda::EqualOp, bool, int32_t, int32_t>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_I64:
        return _device_info->calculate<256, cuda::EqualOp, bool, int64_t, int64_t>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_F64:
        return _device_info->calculate<256, cuda::EqualOp, bool, double, double>(_info, workspace, output, inputs, stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::equal::nvidia
