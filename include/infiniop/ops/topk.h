// Content for topk.h
#ifndef __TOPK_H__
#define __TOPK_H__

#include "common.h"
#include "core/tensor_descriptor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct infiniopTopKDescriptor *infiniopTopKDescriptor_t;

/**
 * @brief Creates a TopK operator descriptor.
 *
 * @param handle The infiniop handle.
 * @param desc The pointer to the TopK descriptor.
 * @param input_desc The tensor descriptor for the input tensor.
 * @param output_val_desc The tensor descriptor for the output values tensor.
 * @param output_ind_desc The tensor descriptor for the output indices tensor.
 * @param k The number of top elements to look for.
 */
void infiniopCreateTopKDescriptor(infiniopHandle_t handle,
                                  infiniopTopKDescriptor_t *desc,
                                  infiniopTensorDescriptor_t input_desc,
                                  infiniopTensorDescriptor_t output_val_desc,
                                  infiniopTensorDescriptor_t output_ind_desc,
                                  int k);

/**
 * @brief Destroys a TopK operator descriptor.
 *
 * @param desc The TopK descriptor to be destroyed.
 */
void infiniopDestroyTopKDescriptor(infiniopTopKDescriptor_t desc);

/**
 * @brief Computes the TopK operation.
 *
 * @param desc The TopK descriptor.
 * @param workspace The workspace memory.
 * @param workspace_size The size of the workspace memory.
 * @param output_val The pointer to the output values tensor.
 * @param output_ind The pointer to the output indices tensor.
 * @param input The pointer to the input tensor.
 * @param stream The CUDA stream.
 */
void infiniopTopK(infiniopTopKDescriptor_t desc, void *workspace,
                  size_t workspace_size, void *output_val, void *output_ind,
                  const void *input, infiniopStream_t stream);

/**
 * @brief Get the workspace size required for the TopK operation.
 *
 * @param desc The TopK descriptor.
 * @param size The pointer to the size of the workspace memory.
 */
void infiniopGetTopKWorkspaceSize(infiniopTopKDescriptor_t desc, size_t *size);

#ifdef __cplusplus
}
#endif

#endif // __TOPK_H__ 