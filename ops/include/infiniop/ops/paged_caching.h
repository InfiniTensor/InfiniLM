#ifndef __INFINIOP_PAGED_CACHING_API_H__
#define __INFINIOP_PAGED_CACHING_API_H__

#include "../operator_descriptor.h"

// Define an opaque handle for the Paged Caching descriptor.
typedef struct InfiniopDescriptor *infiniopPagedCachingDescriptor_t;

/**
 * @brief Creates a descriptor for the Paged Caching operation.
 *
 * This function initializes a descriptor that holds all the metadata needed
 * to copy key/value vectors into their respective cache pools.
 *
 * @param handle The handle to the InfiniOP library context.
 * @param desc_ptr A pointer to store the created descriptor.
 * @param k_cache_desc Descriptor for the key cache pool tensor.
 * @param v_cache_desc Descriptor for the value cache pool tensor.
 * @param k_desc Descriptor for the source key tensor.
 * @param v_desc Descriptor for the source value tensor.
 * @param slot_mapping_desc Descriptor for the slot mapping tensor.
 * @return infiniStatus_t Status code of the operation.
 */
__INFINI_C __export infiniStatus_t infiniopCreatePagedCachingDescriptor(
    infiniopHandle_t handle,
    infiniopPagedCachingDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t k_cache_desc,
    infiniopTensorDescriptor_t v_cache_desc,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t v_desc,
    infiniopTensorDescriptor_t slot_mapping_desc);

/**
 * @brief Retrieves the workspace size required for the Paged Caching operation.
 *
 * @param desc The Paged Caching descriptor.
 * @param size A pointer to store the required workspace size in bytes (typically 0).
 * @return infiniStatus_t Status code of the operation.
 */
__INFINI_C __export infiniStatus_t infiniopGetPagedCachingWorkspaceSize(
    infiniopPagedCachingDescriptor_t desc, size_t *size);

/**
 * @brief Executes the Paged Caching operation.
 *
 * @param desc The Paged Caching descriptor.
 * @param workspace Pointer to the workspace memory.
 * @param workspace_size The size of the workspace.
 * @param k_cache Pointer to the key cache pool data.
 * @param v_cache Pointer to the value cache pool data.
 * @param k Pointer to the source key tensor data.
 * @param v Pointer to the source value tensor data.
 * @param slot_mapping Pointer to the slot mapping data.
 * @param stream The CUDA stream for the operation. Can be NULL.
 * @return infiniStatus_t Status code of the operation.
 */
__INFINI_C __export infiniStatus_t infiniopPagedCaching(
    infiniopPagedCachingDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *k_cache,
    void *v_cache,
    const void *k,
    const void *v,
    const void *slot_mapping,
    void *stream);

/**
 * @brief Destroys a Paged Caching descriptor.
 *
 * @param desc The descriptor to be destroyed.
 * @return infiniStatus_t Status code of the operation.
 */
__INFINI_C __export infiniStatus_t infiniopDestroyPagedCachingDescriptor(
    infiniopPagedCachingDescriptor_t desc);

#endif // __INFINIOP_PAGED_CACHING_API_H__
