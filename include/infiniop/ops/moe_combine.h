#ifndef __MOE_COMBINE_H__
#define __MOE_COMBINE_H__

#include "common.h"
#include "core/tensor_descriptor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct infiniopMoECombineDescriptor *infiniopMoECombineDescriptor_t;

/**
 * @brief Creates a MoE Combine operator descriptor.
 *
 * @param handle The infiniop handle.
 * @param desc The pointer to the MoE Combine descriptor.
 * @param permuted_input_desc The tensor descriptor for the permuted input from the experts.
 * @param gating_weights_desc The tensor descriptor for the top-k gating weights.
 * @param aux_info_desc The auxiliary info tensor from the Dispatch step.
 * @param output_desc The final output tensor, in original token order.
 */
void infiniopCreateMoECombineDescriptor(
    infiniopHandle_t handle,
    infiniopMoECombineDescriptor_t *desc,
    infiniopTensorDescriptor_t permuted_input_desc,
    infiniopTensorDescriptor_t gating_weights_desc,
    infiniopTensorDescriptor_t aux_info_desc,
    infiniopTensorDescriptor_t output_desc);

/**
 * @brief Destroys a MoE Combine operator descriptor.
 */
void infiniopDestroyMoECombineDescriptor(infiniopMoECombineDescriptor_t desc);

/**
 * @brief Computes the MoE Combine operation.
 *
 * @param desc The descriptor.
 * @param output Pointer to the final output tensor.
 * @param permuted_input Pointer to the permuted input from experts.
 * @param gating_weights Pointer to the top-k gating weights.
 * @param aux_info Pointer to the auxiliary info from the Dispatch step.
 * @param stream The CUDA stream.
 */
void infiniopMoECombine(infiniopMoECombineDescriptor_t desc,
                        void *output,
                        const void *permuted_input,
                        const void *gating_weights,
                        const void *aux_info,
                        infiniopStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // __MOE_COMBINE_H__ 