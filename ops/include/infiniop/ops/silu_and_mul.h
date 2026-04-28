#ifndef __INFINIOP_SILU_AND_MUL_API_H__
#define __INFINIOP_SILU_AND_MUL_API_H__

#include "../operator_descriptor.h"

/**
 * @brief Opaque handle for the SiluAndMul descriptor.
 */
typedef struct InfiniopDescriptor *infiniopSiluAndMulDescriptor_t;

/**
 * @brief Creates a descriptor for the SiLU and Multiply (SiluAndMul) operation.
 *
 * Format: (input_shape, output_shape)
 * Referencing vLLM kernel SiluAndMul interface:
 * - input_shape is [..., 2*d]  (last dimension is split into two halves for SiLU and multiplication)
 * - output_shape is [..., d]   (last dimension reduced to half)
 *
 * @param handle The handle to the InfiniOP library context.
 * @param desc_ptr A pointer to store the created descriptor.
 * @param output Descriptor for the output tensor. Shape [..., d].
 * @param input Descriptor for the input tensor. Shape [..., 2*d].
 * @return infiniStatus_t Status code of the operation.
 */
__INFINI_C __export infiniStatus_t infiniopCreateSiluAndMulDescriptor(
    infiniopHandle_t handle,
    infiniopSiluAndMulDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output,
    infiniopTensorDescriptor_t input);

/**
 * @brief Queries the workspace size required for SiluAndMul computation.
 * @param desc The SiluAndMul descriptor.
 * @param size Pointer to store the required workspace size in bytes.
 * @return infiniStatus_t Status code of the operation.
 */
__INFINI_C __export infiniStatus_t infiniopGetSiluAndMulWorkspaceSize(
    infiniopSiluAndMulDescriptor_t desc,
    size_t *size);

/**
 * @brief Executes the SiluAndMul operation.
 *
 * Performs SiLU activation on the first half of the last dimension of `input`,
 * multiplies element-wise with the second half, and stores the result in `output`.
 *
 * @param desc The SiluAndMul descriptor.
 * @param workspace Pointer to workspace memory allocated according to GetWorkspaceSize().
 * @param workspace_size Size of the workspace in bytes.
 * @param output Pointer to the output tensor memory. Shape [..., d].
 * @param input Pointer to the input tensor memory. Shape [..., 2*d].
 * @param stream Pointer to the execution stream (e.g., CUDA stream). Can be NULL for default stream.
 * @return infiniStatus_t Status code of the operation.
 */
__INFINI_C __export infiniStatus_t infiniopSiluAndMul(
    infiniopSiluAndMulDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream);

/**
 * @brief Destroys a previously created SiluAndMul descriptor.
 * @param desc The descriptor to destroy.
 * @return infiniStatus_t Status code of the operation.
 */
__INFINI_C __export infiniStatus_t infiniopDestroySiluAndMulDescriptor(
    infiniopSiluAndMulDescriptor_t desc);

#endif // __INFINIOP_SILU_AND_MUL_API_H__
