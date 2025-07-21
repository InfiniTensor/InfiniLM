#include "moe_combine_info.h"
#include "utils/data_type.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace infiniop {

void MoECombineInfo::op(void *output, const void *permuted_input,
                      const void *gating_weights, const void *aux_info,
                      cudaStream_t stream) const {
    moe_combine_kernel_launcher(permuted_input, gating_weights, aux_info,
                                output, this->num_tokens, this->k,
                                this->hidden_dim, this->data_type, stream);
}

template <typename T>
__global__ void combine_kernel(const T *permuted_input,
                               const T *gating_weights,
                               const int *aux_info,
                               T *output,
                               int num_tokens, int k, int hidden_dim) {
    int dispatch_pos = blockIdx.x;
    int data_idx = threadIdx.x;

    if (dispatch_pos >= num_tokens * k || data_idx >= hidden_dim) return;

    // Retrieve the original position and the position of the gating weight
    int original_token_pos = aux_info[dispatch_pos * 2 + 0];
    int gating_val_pos = aux_info[dispatch_pos * 2 + 1];

    // Get the gating weight
    T weight = gating_weights[gating_val_pos];

    // Get the data from the expert's output
    T data = permuted_input[dispatch_pos * hidden_dim + data_idx];

    // Atomically add the weighted result to the final output tensor
    // This handles the case where k > 1 and multiple experts contribute to the same token.
    atomicAdd(&output[original_token_pos * hidden_dim + data_idx], data * weight);
}

void moe_combine_kernel_launcher(const void *permuted_input,
                                 const void *gating_weights,
                                 const void *aux_info, void *output,
                                 int num_tokens, int k, int hidden_dim,
                                 DataType data_type, cudaStream_t stream) {
    if (data_type != DataType::F32) {
        IT_TODO_HALT_MSG("Unsupported data type for MoE Combine");
    }
    
    // The output tensor should be zero-initialized before this kernel.
    // Assuming the framework handles this. If not, a cudaMemsetAsync is needed here.
    // cudaMemsetAsync(output, 0, num_tokens * hidden_dim * sizeof(float), stream);


    dim3 grid(num_tokens * k);
    dim3 block(hidden_dim);
    if (hidden_dim > 1024) block.x = 1024;

    combine_kernel<float><<<grid, block, 0, stream>>>(
        (const float*)permuted_input,
        (const float*)gating_weights,
        (const int*)aux_info,
        (float*)output,
        num_tokens, k, hidden_dim);
}

} // namespace infiniop 