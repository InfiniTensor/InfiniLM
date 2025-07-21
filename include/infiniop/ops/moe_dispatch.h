#ifndef __MOE_DISPATCH_H__
#define __MOE_DISPATCH_H__

#include "common.h"
#include "core/tensor_descriptor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct infiniopMoEDispatchDescriptor *infiniopMoEDispatchDescriptor_t;

/**
 * @brief Creates a MoE Dispatch operator descriptor.
 *
 * @param handle The infiniop handle.
 * @param desc The pointer to the MoE Dispatch descriptor.
 * @param num_experts The total number of experts.
 * @param input_desc The tensor descriptor for the input hidden states.
 * @param indices_desc The tensor descriptor for the top-k indices from the gating network.
 * @param permuted_output_desc The tensor descriptor for the permuted (dispatched) output.
 * @param aux_info_desc An auxiliary output to store permutation info needed by the Combine step.
 */
void infiniopCreateMoEDispatchDescriptor(
    infiniopHandle_t handle,
    infiniopMoEDispatchDescriptor_t *desc,
    int num_experts,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t indices_desc,
    infiniopTensorDescriptor_t permuted_output_desc,
    infiniopTensorDescriptor_t aux_info_desc);

/**
 * @brief Destroys a MoE Dispatch operator descriptor.
 */
void infiniopDestroyMoEDispatchDescriptor(infiniopMoEDispatchDescriptor_t desc);

/**
 * @brief Computes the MoE Dispatch operation.
 *
 * @param desc The descriptor.
 * @param permuted_output Pointer to the permuted output tensor.
 * @param aux_info Pointer to the auxiliary info tensor.
 * @param input Pointer to the input hidden states.
 * @param indices Pointer to the top-k indices.
 * @param stream The CUDA stream.
 */
void infiniopMoEDispatch(infiniopMoEDispatchDescriptor_t desc,
                       void *permuted_output,
                       void *aux_info,
                       const void *input,
                       const void *indices,
                       infiniopStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // __MOE_DISPATCH_H__ 