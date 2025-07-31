#ifndef __INFINIOP_MOE_EXPERT_INFO_API_H__
#define __INFINIOP_MOE_EXPERT_INFO_API_H__

#include "../operator_descriptor.h"

/**
 * @brief An opaque handle to a MoEExpertInfo operator descriptor.
 * This descriptor stores pre-calculated metadata for the operation.
 */
typedef struct InfiniopDescriptor *infiniopMoEExpertInfoDescriptor_t;

/**
 * @brief Creates a descriptor for the MoE expert info calculation.
 *
 * This function validates the tensor descriptors and initializes a descriptor
 * for the operation that counts token assignments per expert and calculates their offsets.
 *
 * @param handle The library handle.
 * @param desc Pointer to the descriptor to be created.
 * @param topk_ind_desc Descriptor for the input tensor of expert indices.
 * @param expert_counts_desc Descriptor for the output tensor for expert counts.
 * @param expert_offsets_desc Descriptor for the output tensor for expert offsets.
 * @return `infiniStatus_t` indicating success or failure.
 */
__C __export infiniStatus_t infiniopCreateMoEExpertInfoDescriptor(
    infiniopHandle_t handle,
    infiniopMoEExpertInfoDescriptor_t *desc,
    infiniopTensorDescriptor_t topk_ind_desc,
    infiniopTensorDescriptor_t expert_counts_desc,
    infiniopTensorDescriptor_t expert_offsets_desc);

/**
 * @brief Destroys a MoEExpertInfo descriptor.
 *
 * @param desc The descriptor to be destroyed.
 * @return `infiniStatus_t` indicating success or failure.
 */
__C __export infiniStatus_t infiniopDestroyMoEExpertInfoDescriptor(infiniopMoEExpertInfoDescriptor_t desc);

/**
 * @brief Executes the MoE expert info calculation.
 *
 * This function launches the CUDA kernels to populate the counts and offsets buffers.
 *
 * @param handle The library handle.
 * @param desc The descriptor for this operation.
 * @param topk_ind Pointer to the input device memory for expert indices.
 * @param expert_counts Pointer to the output device memory for expert counts.
 * @param expert_offsets Pointer to the output device memory for expert offsets.
 * @param stream The CUDA stream on which to execute the kernels.
 * @return `infiniStatus_t` indicating success or failure.
 */
__C __export infiniStatus_t infiniopMoEExpertInfoCalculate(
    infiniopHandle_t handle,
    infiniopMoEExpertInfoDescriptor_t desc,
    const void *topk_ind,
    void *expert_counts,
    void *expert_offsets,
    infinirtStream_t stream);

#endif // __INFINIOP_MOE_EXPERT_INFO_API_H__