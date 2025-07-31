#include "moe_expert_info.cuh"
#include <cuda_runtime.h>
#include "../../../devices/nvidia/nvidia_handle.cuh"
#include <cub/device/device_scan.cuh>
#include "../info.h"
#include <cuda_fp16.h>

namespace op::moe_expert_info::nvidia {

struct Descriptor::Opaque {
	std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
	delete _opaque;
}
//=========== CUDA KERNELS ===========//

/**
 * @brief CUDA Kernel to count token assignments for each expert.
 * It iterates through the TopK indices and atomically increments a counter
 * for each expert.
 */
__global__ void count_expert_assignments_kernel(int *expert_counts,
                                                const int *topk_indices,
                                                int num_tokens, int top_k) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < num_tokens; i += stride) {
        for (int k = 0; k < top_k; ++k) {
            int expert_idx = topk_indices[i * top_k + k];
            // Atomically add to the counter for the assigned expert
            atomicAdd(&expert_counts[expert_idx], 1);
        }
    }
}

/**
 * @brief CUDA Kernel to perform a simple exclusive scan.
 * It calculates the starting offset for each expert's data block.
 * Note: This is a simple, single-threaded implementation suitable for a small number of experts.
 */
__global__ void exclusive_scan_kernel(int *expert_offsets,
                                      const int *expert_counts,
                                      int num_experts) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int total = 0;
        for (int i = 0; i < num_experts; ++i) {
            expert_offsets[i] = total;
            total += expert_counts[i];
        }
    }
}

//=========== DESCRIPTOR METHOD IMPLEMENTATIONS ===========//

/**
 * @brief Factory method to create a descriptor instance.
 * It first validates tensor shapes and types using MoEExpertInfoInfo.
 */
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t topk_ind_desc,
    infiniopTensorDescriptor_t expert_counts_desc,
    infiniopTensorDescriptor_t expert_offsets_desc) {

    // Validate inputs using the Info class
    auto info_res = MoEExpertInfoInfo::create(topk_ind_desc, expert_counts_desc, expert_offsets_desc);
	CHECK_RESULT(info_res);
    
    // Allocate opaque struct and the descriptor itself
    auto opaque = new Opaque{};
    auto desc = new Descriptor(*info_res, opaque, handle->device, handle->device_id);
    *desc_ptr = desc;

    return INFINI_STATUS_SUCCESS;
}


/**
 * @brief Executes the operator's computation.
 * This method launches the CUDA kernels on the specified stream.
 */
infiniStatus_t Descriptor::calculate(const void *topk_ind, void *expert_counts,
                                     void *expert_offsets,
                                     void *stream) const {
    auto num_tokens = _info.num_tokens;
    auto num_experts = _info.num_experts;
    auto top_k = _info.k;
    auto cuda_stream = static_cast<cudaStream_t>(stream);

    // 1. Reset the counts buffer to zero before counting.
    cudaMemsetAsync(expert_counts, 0, sizeof(int) * num_experts, cuda_stream);

    // 2. Launch the kernel to count expert assignments.
    int threads_per_block = 256;
    int blocks = (num_tokens + threads_per_block - 1) / threads_per_block;
    count_expert_assignments_kernel<<<blocks, threads_per_block, 0, cuda_stream>>>(
        (int *)expert_counts, (const int *)topk_ind, num_tokens, top_k);

    // 3. Launch the kernel to calculate offsets via exclusive scan.
    exclusive_scan_kernel<<<1, 1, 0, cuda_stream>>>(
        (int *)expert_offsets, (const int *)expert_counts, num_experts);

    // In a production library, a macro would typically be used for robust error checking.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // You might want to log the error string here: cudaGetErrorString(err)
        return INFINI_STATUS_NULL_POINTER;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::moe_expert_info::nvidia