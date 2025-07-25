#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "moe_combine.cuh"

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

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    auto result = MoECombineInfo::create(
        permuted_input_desc, gating_weights_desc, aux_info_desc, output_desc);
    CHECK_RESULT(result);


	*desc_ptr = new Descriptor(dtype, result.take(), 0,
	 new Opaque{handle->internal()}, handle->device,
                               handle_->device_id);
    return INFINI_STATUS_SUCCESS;
}


template <typename T>
__global__ void kernel(const T *permuted_input, const T *gating_weights,
                     const int *aux_info, T *output, int num_tokens, int k,
                     int hidden_dim) {
    int dispatch_pos = blockIdx.x * blockDim.x + threadIdx.x;
    int data_idx = threadIdx.y;
    int max_threads = blockDim.x * blockDim.y;

    for (int i = dispatch_pos * max_threads + data_idx;
         i < num_tokens * k * hidden_dim; i += gridDim.x * max_threads) {
        int current_dispatch_pos = i / hidden_dim;
        int current_data_idx = i % hidden_dim;
        int original_token_pos = aux_info[current_dispatch_pos * 2 + 0];
        int gating_val_pos = aux_info[current_dispatch_pos * 2 + 1];
        T weight = gating_weights[gating_val_pos];
        T data = permuted_input[i];
        atomicAdd(&output[original_token_pos * hidden_dim + current_data_idx],
                  data * weight);
    }
}
infiniStatus_t
Descriptor::calculate(const void *permuted_input, const void *gating_weights,
                    const void *aux_info, void *output, void *stream) const {
    if (_info.data_type() != INFINI_DTYPE_F32) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    cudaMemsetAsync(output, 0,
                    _info.num_tokens() * _info.hidden_dim() * sizeof(float),
                    (cudaStream_t)stream);

    dim3 grid(256);
    dim3 block(32, 32);
    kernel<float><<<grid, block, 0, (cudaStream_t)stream>>>(
        (const float *)permuted_input, (const float *)gating_weights,
        (const int *)aux_info, (float *)output, _info.num_tokens(), _info.k(),
        _info.hidden_dim());

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::moe_combine::cuda 