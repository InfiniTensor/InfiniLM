#include "linear_nvidia.cuh"
#include "../../../devices/nvidia/common_nvidia.cuh"

namespace op::linear::nvidia {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t bias_desc) {
    
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto dtype = output_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    // Check that input and weight have the same dtype
    if (input_desc->dtype() != dtype || weight_desc->dtype() != dtype) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    // Check bias dtype if provided
    if (bias_desc && bias_desc->dtype() != dtype) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    auto result = LinearInfo::create(output_desc, input_desc, weight_desc, bias_desc);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        dtype, result.take(), 0,
        nullptr,
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata>
__global__ void linear_kernel(
    Tdata *output,
    const Tdata *input,
    const Tdata *weight,
    const Tdata *bias,
    size_t batch_size,
    size_t in_features,
    size_t out_features,
    bool has_bias,
    ptrdiff_t input_batch_stride,
    ptrdiff_t input_feature_stride,
    ptrdiff_t output_batch_stride,
    ptrdiff_t output_feature_stride,
    ptrdiff_t weight_out_stride,
    ptrdiff_t weight_in_stride,
    ptrdiff_t bias_stride) {

    size_t batch_idx = blockIdx.x;
    size_t out_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (batch_idx >= batch_size || out_idx >= out_features) {
        return;
    }

    float sum = 0.0f;
    
    // Compute dot product: input[batch_idx, :] * weight[out_idx, :]
    for (size_t in_idx = threadIdx.x; in_idx < in_features; in_idx += blockDim.x) {
        auto input_offset = batch_idx * input_batch_stride + in_idx * input_feature_stride;
        auto weight_offset = out_idx * weight_out_stride + in_idx * weight_in_stride;
        
        if constexpr (std::is_same<Tdata, half>::value || std::is_same<Tdata, __nv_bfloat16>::value) {
            sum += __half2float(input[input_offset]) * __half2float(weight[weight_offset]);
        } else {
            sum += input[input_offset] * weight[weight_offset];
        }
    }

    // Reduce within warp
    for (int mask = 16; mask > 0; mask /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, mask);
    }

    // First thread in warp writes result
    if (threadIdx.x == 0) {
        // Add bias if provided
        if (has_bias) {
            if constexpr (std::is_same<Tdata, half>::value || std::is_same<Tdata, __nv_bfloat16>::value) {
                sum += __half2float(bias[out_idx * bias_stride]);
            } else {
                sum += bias[out_idx * bias_stride];
            }
        }
        
        auto output_offset = batch_idx * output_batch_stride + out_idx * output_feature_stride;
        if constexpr (std::is_same<Tdata, half>::value || std::is_same<Tdata, __nv_bfloat16>::value) {
            output[output_offset] = __float2half(sum);
        } else {
            output[output_offset] = sum;
        }
    }
}

template <typename Tdata>
void calculate(
    const LinearInfo &info,
    void *output,
    const void *input,
    const void *weight,
    const void *bias,
    cudaStream_t stream) {

    auto output_ptr = reinterpret_cast<Tdata *>(output);
    auto input_ptr = reinterpret_cast<const Tdata *>(input);
    auto weight_ptr = reinterpret_cast<const Tdata *>(weight);
    auto bias_ptr = info.has_bias ? reinterpret_cast<const Tdata *>(bias) : nullptr;

    // Configure grid and block dimensions
    constexpr int WARP_SIZE = 32;
    constexpr int BLOCK_SIZE_Y = 8;
    
    dim3 blockDim(WARP_SIZE, BLOCK_SIZE_Y);
    dim3 gridDim(info.batch_size, (info.out_features + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);

    linear_kernel<<<gridDim, blockDim, 0, stream>>>(
        output_ptr, input_ptr, weight_ptr, bias_ptr,
        info.batch_size, info.in_features, info.out_features,
        info.has_bias,
        info.input_batch_stride, info.input_feature_stride,
        info.output_batch_stride, info.output_feature_stride,
        info.weight_out_stride, info.weight_in_stride,
        info.bias_stride);
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    const void *weight,
    const void *bias,
    void *stream) const {

    auto cuda_stream = static_cast<cudaStream_t>(stream);

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        nvidia::calculate<half>(_info, output, input, weight, bias, cuda_stream);
        return INFINI_STATUS_SUCCESS;

    case INFINI_DTYPE_BF16:
        nvidia::calculate<__nv_bfloat16>(_info, output, input, weight, bias, cuda_stream);
        return INFINI_STATUS_SUCCESS;

    case INFINI_DTYPE_F32:
        nvidia::calculate<float>(_info, output, input, weight, bias, cuda_stream);
        return INFINI_STATUS_SUCCESS;

    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::linear::nvidia