#ifndef __INFINIOP_GQA_API_H__
#define __INFINIOP_GQA_API_H__

#include "../operator_descriptor.h" // Assumes a base header for descriptors


// Opaque pointer type for the GQA descriptor to hide implementation details.
typedef struct InfiniopDescriptor *infiniopGQADescriptor_t;

/**
 * @brief Creates a descriptor for the Grouped-Query Attention (GQA) operation.
 *
 * @param handle The library handle.
 * @param desc A pointer to the descriptor to be created.
 * @param q_desc The tensor descriptor for the Query (Q) input.
 * @param k_desc The tensor descriptor for the Key (K) input.
 * @param v_desc The tensor descriptor for the Value (V) input.
 * @param output_desc The tensor descriptor for the output.
 * @return An infiniStatus_t indicating success or failure.
 */
__C __export infiniStatus_t infiniopCreateGQADescriptor(
    infiniopHandle_t handle,
    infiniopGQADescriptor_t *desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t v_desc,
    infiniopTensorDescriptor_t output_desc);

/**
 * @brief Destroys a GQA descriptor and releases its resources.
 *
 * @param desc The descriptor to destroy.
 * @return An infiniStatus_t indicating success or failure.
 */
__C __export infiniStatus_t infiniopDestroyGQADescriptor(infiniopGQADescriptor_t desc);

/**
 * @brief Executes the Grouped-Query Attention (GQA) operation.
 *
 * @param desc The GQA descriptor.
 * @param q Pointer to the Query (Q) input data on the device.
 * @param k Pointer to the Key (K) input data on the device.
 * @param v Pointer to the Value (V) input data on the device.
 * @param output Pointer to the output tensor data on the device.
 * @param stream A pointer to the CUDA/HIP stream for asynchronous execution.
 * @return An infiniStatus_t indicating success or failure.
 */
__C __export infiniStatus_t infiniopGQA(infiniopGQADescriptor_t desc,
                                     const void *q,
                                     const void *k,
                                     const void *v,
                                     void *output,
                                     void *stream);

#endif // __INFINIOP_GQA_API_H__
