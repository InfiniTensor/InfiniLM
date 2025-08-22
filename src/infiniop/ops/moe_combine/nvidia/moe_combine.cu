#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "moe_combine.cuh"
#include <cuda_fp16.h>   // For __half (FP16)
#include <cuda_bf16.h>  // For __nv_bfloat16 (BF16)

namespace op::moe_combine::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_, Descriptor **desc_ptr,
    infiniopTensorDescriptor_t permuted_input_desc,
    infiniopTensorDescriptor_t gating_weights_desc,
    infiniopTensorDescriptor_t aux_info_desc,
    infiniopTensorDescriptor_t output_desc) {

    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto dtype = permuted_input_desc->dtype();

    // The check already includes the desired types.
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);
	cudaDeviceProp prop;
    // handle->device 存储了当前使用的设备ID
    cudaGetDeviceProperties(&prop, handle->device);

    // 检查FP16支持。FP16的atomicAdd需要计算能力7.0及以上的GPU (Volta架构及更新)
    if (dtype == INFINI_DTYPE_F16 && prop.major < 7) {
        // 如果硬件不支持，则返回架构不匹配的错误
        // TODO: 可以添加日志记录更详细的错误信息
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

    // 检查BF16支持。BF16的atomicAdd需要计算能力8.0及以上的GPU (Ampere架构及更新)
    if (dtype == INFINI_DTYPE_BF16 && prop.major < 8) {
        // 如果硬件不支持，则返回架构不匹配的错误
        // TODO: 可以添加日志记录更详细的错误信息
		printf("YOUR_DEVICE_TYPE_NOT_SUPPORT_BF16,PLEASE_CHECK_YOUR_GPU_ARCH\n");
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
    auto result = MoECombineInfo::create(
        permuted_input_desc, gating_weights_desc, aux_info_desc, output_desc);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(dtype, result.take(), 0,
                               new Opaque{handle->internal()}, handle->device,
                               handle_->device_id);
    return INFINI_STATUS_SUCCESS;
}

/**
 * @brief CUDA kernel to combine expert outputs.
 *
 * This kernel performs a weighted sum of permuted expert outputs. It reads
 * the permuted input data, the corresponding gating weights, and auxiliary
 * information to scatter the results back to their original token positions.
 *
 * @tparam T The data type of the tensors (float, half, or __nv_bfloat16).
 * @param permuted_input Pointer to the input tensor after expert permutation.
 * @param gating_weights Pointer to the gating weights for each expert output.
 * @param aux_info Auxiliary info containing original token and gating positions.
 * @param output Pointer to the output tensor.
 * @param num_tokens The number of tokens in the batch.
 * @param k The number of experts each token is routed to (top-k).
 * @param hidden_dim The hidden dimension size of the model.
 *
 * @note This kernel relies on atomicAdd. Support for half and __nv_bfloat16
 * is available on GPUs with compute capability 7.x and 8.x, respectively.
 * Ensure the code is compiled for the appropriate architecture.
 */
template <typename T>
__global__ void kernel(const T *permuted_input, const T *gating_weights,
                       const int *aux_info, T *output, int num_tokens, int k,
                       int hidden_dim) {
    // Calculate the global thread index for the dispatch position.
    int dispatch_pos = blockIdx.x * blockDim.x + threadIdx.x;
    // Get the thread index within the hidden dimension.
    int data_idx = threadIdx.y;
    // Calculate the total number of threads in the y-dimension of the block.
    int max_threads = blockDim.x * blockDim.y;

    // Grid-stride loop to ensure all elements are processed.
    for (int i = dispatch_pos * max_threads + data_idx;
         i < num_tokens * k * hidden_dim; i += gridDim.x * max_threads) {
        // Deconstruct the linear index 'i' to get metadata.
        int current_dispatch_pos = i / hidden_dim;
        int current_data_idx = i % hidden_dim;

        // Retrieve the original token position and gating value position from aux_info.
        // aux_info is structured as [original_token_pos, gating_val_pos, ...]
        int original_token_pos = aux_info[current_dispatch_pos * 2 + 0];
        int gating_val_pos = aux_info[current_dispatch_pos * 2 + 1];

        // Fetch the gating weight and the permuted data.
        T weight = gating_weights[gating_val_pos];
        T data = permuted_input[i];

        // Atomically add the weighted data to the correct position in the output tensor.
        // The multiplication (data * weight) is promoted to FP32 for precision
        // before being converted back to T for the atomic addition.
        atomicAdd(&output[original_token_pos * hidden_dim + current_data_idx],
                  data * weight);
    }
}

// Helper function to launch the templated kernel
template <typename T>
void launch_kernel(const void *permuted_input, const void *gating_weights,
                   const void *aux_info, void *output,
                   const MoECombineInfo &_info, cudaStream_t stream) {
    size_t output_bytes = _info.num_tokens * _info.hidden_dim * sizeof(T);
    cudaMemsetAsync(output, 0, output_bytes, stream);

    dim3 grid(256);
    dim3 block(32, 32);
    kernel<T><<<grid, block, 0, stream>>>(
        static_cast<const T *>(permuted_input),
        static_cast<const T *>(gating_weights),
        static_cast<const int *>(aux_info), static_cast<T *>(output),
        _info.num_tokens, _info.k, _info.hidden_dim);
}

infiniStatus_t
Descriptor::calculate(const void *permuted_input, const void *gating_weights,
                      const void *aux_info, void *output, void *stream) const {
    // Dispatch to the correct kernel based on the data type.
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    switch (_info.data_type) {
    case INFINI_DTYPE_F32:
        launch_kernel<float>(permuted_input, gating_weights, aux_info, output,
                             _info, cuda_stream);
        break;
    case INFINI_DTYPE_F16:
        launch_kernel<half>(permuted_input, gating_weights, aux_info, output,
                            _info, cuda_stream);
        break;
    case INFINI_DTYPE_BF16:
        launch_kernel<__nv_bfloat16>(permuted_input, gating_weights, aux_info,
                                     output, _info, cuda_stream);
        break;
    default:
        // Handle unsupported data types.
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::moe_combine::nvidia
