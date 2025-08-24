#include "linear_backwards_nvidia.cuh"
#include "../../../devices/nvidia/common_nvidia.cuh"

namespace op::linear_backwards::nvidia {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t grad_input_desc,
    infiniopTensorDescriptor_t grad_weight_desc,
    infiniopTensorDescriptor_t grad_bias_desc,
    infiniopTensorDescriptor_t grad_output_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t weight_desc) {
    
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto dtype = grad_output_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    // Check that all tensors have the same dtype
    if (input_desc->dtype() != dtype || weight_desc->dtype() != dtype) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    if (grad_input_desc && grad_input_desc->dtype() != dtype) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    if (grad_weight_desc && grad_weight_desc->dtype() != dtype) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    if (grad_bias_desc && grad_bias_desc->dtype() != dtype) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    auto result = LinearBackwardsInfo::create(grad_input_desc, grad_weight_desc, grad_bias_desc,
                                              grad_output_desc, input_desc, weight_desc);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        dtype, result.take(), 0,
        nullptr,
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

// Kernel for computing grad_input = grad_output @ weight
template <typename Tdata>
__global__ void grad_input_kernel(
    Tdata *grad_input,
    const Tdata *grad_output,
    const Tdata *weight,
    size_t batch_size,
    size_t in_features,
    size_t out_features,
    ptrdiff_t grad_input_batch_stride,
    ptrdiff_t grad_input_feature_stride,
    ptrdiff_t grad_output_batch_stride,
    ptrdiff_t grad_output_feature_stride,
    ptrdiff_t weight_out_stride,
    ptrdiff_t weight_in_stride) {

    size_t batch_idx = blockIdx.x;
    size_t in_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (batch_idx >= batch_size || in_idx >= in_features) {
        return;
    }

    float sum = 0.0f;
    
    // grad_input[batch_idx, in_idx] = sum(grad_output[batch_idx, :] * weight[:, in_idx])
    for (size_t out_idx = threadIdx.x; out_idx < out_features; out_idx += blockDim.x) {
        auto grad_output_offset = batch_idx * grad_output_batch_stride + out_idx * grad_output_feature_stride;
        auto weight_offset = out_idx * weight_out_stride + in_idx * weight_in_stride;
        
        if constexpr (std::is_same<Tdata, half>::value || std::is_same<Tdata, __nv_bfloat16>::value) {
            sum += __half2float(grad_output[grad_output_offset]) * __half2float(weight[weight_offset]);
        } else {
            sum += grad_output[grad_output_offset] * weight[weight_offset];
        }
    }

    // Reduce within warp
    for (int mask = 16; mask > 0; mask /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, mask);
    }

    // First thread in warp writes result
    if (threadIdx.x == 0) {
        auto grad_input_offset = batch_idx * grad_input_batch_stride + in_idx * grad_input_feature_stride;
        if constexpr (std::is_same<Tdata, half>::value || std::is_same<Tdata, __nv_bfloat16>::value) {
            grad_input[grad_input_offset] = __float2half(sum);
        } else {
            grad_input[grad_input_offset] = sum;
        }
    }
}

// Kernel for computing grad_weight = grad_output.T @ input
template <typename Tdata>
__global__ void grad_weight_kernel(
    Tdata *grad_weight,
    const Tdata *grad_output,
    const Tdata *input,
    size_t batch_size,
    size_t in_features,
    size_t out_features,
    ptrdiff_t grad_weight_out_stride,
    ptrdiff_t grad_weight_in_stride,
    ptrdiff_t grad_output_batch_stride,
    ptrdiff_t grad_output_feature_stride,
    ptrdiff_t input_batch_stride,
    ptrdiff_t input_feature_stride) {

    size_t out_idx = blockIdx.x;
    size_t in_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (out_idx >= out_features || in_idx >= in_features) {
        return;
    }

    float sum = 0.0f;
    
    // grad_weight[out_idx, in_idx] = sum(grad_output[:, out_idx] * input[:, in_idx])
    for (size_t batch_idx = threadIdx.x; batch_idx < batch_size; batch_idx += blockDim.x) {
        auto grad_output_offset = batch_idx * grad_output_batch_stride + out_idx * grad_output_feature_stride;
        auto input_offset = batch_idx * input_batch_stride + in_idx * input_feature_stride;
        
        if constexpr (std::is_same<Tdata, half>::value || std::is_same<Tdata, __nv_bfloat16>::value) {
            sum += __half2float(grad_output[grad_output_offset]) * __half2float(input[input_offset]);
        } else {
            sum += grad_output[grad_output_offset] * input[input_offset];
        }
    }

    // Reduce within warp
    for (int mask = 16; mask > 0; mask /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, mask);
    }

    // First thread in warp writes result
    if (threadIdx.x == 0) {
        auto grad_weight_offset = out_idx * grad_weight_out_stride + in_idx * grad_weight_in_stride;
        if constexpr (std::is_same<Tdata, half>::value || std::is_same<Tdata, __nv_bfloat16>::value) {
            grad_weight[grad_weight_offset] = __float2half(sum);
        } else {
            grad_weight[grad_weight_offset] = sum;
        }
    }
}

// Kernel for computing grad_bias = sum(grad_output, dim=0)
template <typename Tdata>
__global__ void grad_bias_kernel(
    Tdata *grad_bias,
    const Tdata *grad_output,
    size_t batch_size,
    size_t out_features,
    ptrdiff_t grad_bias_stride,
    ptrdiff_t grad_output_batch_stride,
    ptrdiff_t grad_output_feature_stride) {

    size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (out_idx >= out_features) {
        return;
    }

    float sum = 0.0f;
    
    // grad_bias[out_idx] = sum(grad_output[:, out_idx])
    for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        auto grad_output_offset = batch_idx * grad_output_batch_stride + out_idx * grad_output_feature_stride;
        
        if constexpr (std::is_same<Tdata, half>::value || std::is_same<Tdata, __nv_bfloat16>::value) {
            sum += __half2float(grad_output[grad_output_offset]);
        } else {
            sum += grad_output[grad_output_offset];
        }
    }
    
    auto grad_bias_offset = out_idx * grad_bias_stride;
    if constexpr (std::is_same<Tdata, half>::value || std::is_same<Tdata, __nv_bfloat16>::value) {
        grad_bias[grad_bias_offset] = __float2half(sum);
    } else {
        grad_bias[grad_bias_offset] = sum;
    }
}

template <typename Tdata>
void calculate(
    const LinearBackwardsInfo &info,
    void *grad_input,
    void *grad_weight,
    void *grad_bias,
    const void *grad_output,
    const void *input,
    const void *weight,
    cudaStream_t stream) {

    auto grad_output_ptr = reinterpret_cast<const Tdata *>(grad_output);
    auto input_ptr = reinterpret_cast<const Tdata *>(input);
    auto weight_ptr = reinterpret_cast<const Tdata *>(weight);

    auto grad_input_ptr = grad_input ? reinterpret_cast<Tdata *>(grad_input) : nullptr;
    auto grad_weight_ptr = grad_weight ? reinterpret_cast<Tdata *>(grad_weight) : nullptr;
    auto grad_bias_ptr = grad_bias ? reinterpret_cast<Tdata *>(grad_bias) : nullptr;

    constexpr int WARP_SIZE = 32;
    constexpr int BLOCK_SIZE_Y = 8;

    // Compute grad_input = grad_output @ weight
    if (grad_input_ptr) {
        dim3 blockDim(WARP_SIZE, BLOCK_SIZE_Y);
        dim3 gridDim(info.batch_size, (info.in_features + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);

        grad_input_kernel<<<gridDim, blockDim, 0, stream>>>(
            grad_input_ptr, grad_output_ptr, weight_ptr,
            info.batch_size, info.in_features, info.out_features,
            info.grad_input_batch_stride, info.grad_input_feature_stride,
            info.grad_output_batch_stride, info.grad_output_feature_stride,
            info.weight_out_stride, info.weight_in_stride);
    }

    // Compute grad_weight = grad_output.T @ input
    if (grad_weight_ptr) {
        dim3 blockDim(WARP_SIZE, BLOCK_SIZE_Y);
        dim3 gridDim(info.out_features, (info.in_features + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);

        grad_weight_kernel<<<gridDim, blockDim, 0, stream>>>(
            grad_weight_ptr, grad_output_ptr, input_ptr,
            info.batch_size, info.in_features, info.out_features,
            info.grad_weight_out_stride, info.grad_weight_in_stride,
            info.grad_output_batch_stride, info.grad_output_feature_stride,
            info.input_batch_stride, info.input_feature_stride);
    }

    // Compute grad_bias = sum(grad_output, dim=0)
    if (grad_bias_ptr && info.has_bias) {
        dim3 blockDim(256);
        dim3 gridDim((info.out_features + 255) / 256);

        grad_bias_kernel<<<gridDim, blockDim, 0, stream>>>(
            grad_bias_ptr, grad_output_ptr,
            info.batch_size, info.out_features,
            info.grad_bias_stride,
            info.grad_output_batch_stride, info.grad_output_feature_stride);
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *grad_input,
    void *grad_weight,
    void *grad_bias,
    const void *grad_output,
    const void *input,
    const void *weight,
    void *stream) const {

    auto cuda_stream = static_cast<cudaStream_t>(stream);

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        nvidia::calculate<half>(_info, grad_input, grad_weight, grad_bias, grad_output, input, weight, cuda_stream);
        return INFINI_STATUS_SUCCESS;

    case INFINI_DTYPE_BF16:
        nvidia::calculate<__nv_bfloat16>(_info, grad_input, grad_weight, grad_bias, grad_output, input, weight, cuda_stream);
        return INFINI_STATUS_SUCCESS;

    case INFINI_DTYPE_F32:
        nvidia::calculate<float>(_info, grad_input, grad_weight, grad_bias, grad_output, input, weight, cuda_stream);
        return INFINI_STATUS_SUCCESS;

    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::linear_backwards::nvidia