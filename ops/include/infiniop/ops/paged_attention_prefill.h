#ifndef __INFINIOP_PAGED_ATTENTION_PREFILL_API_H__
#define __INFINIOP_PAGED_ATTENTION_PREFILL_API_H__

#include "../operator_descriptor.h"

// Define an opaque handle for the Paged Attention Prefill descriptor.
typedef struct InfiniopDescriptor *infiniopPagedAttentionPrefillDescriptor_t;

/**
 * @brief Creates a descriptor for the Paged Attention Prefill operation.
 * @param handle The handle to the InfiniOP library context.
 * @param desc_ptr A pointer to store the created descriptor.
 * @param out_desc Descriptor for the output tensor.
 * Shape: [total_q_tokens, num_heads, head_size]
 * @param q_desc Descriptor for the query tensor (packed/flattened).
 * Shape: [total_q_tokens, num_heads, head_size]
 * @param k_cache_desc Descriptor for the global physical key cache.
 * Shape: [max_num_blocks, num_kv_heads, block_size, head_size]
 * @param v_cache_desc Descriptor for the global physical value cache.
 * Shape: [max_num_blocks, num_kv_heads, block_size, head_size]
 * @param block_tables_desc Descriptor for the block tables mapping logic to physical blocks.
 * Shape: [batch_size, max_blocks_per_seq]
 * @param seq_lens_desc Descriptor for the total KV lengths of each sequence.
 * Shape: [batch_size]
 * @param cum_seq_lens_q_desc Descriptor for the cumulative start position (prefix sum) of each Q sequence.
 * Shape: [batch_size + 1]
 * @param alibi_slopes_desc Optional descriptor for the ALiBi slopes tensor. Can be NULL.
 * Shape: [num_heads]
 * @param scale The attention scaling factor (typically 1.0 / sqrt(head_size)).
 * @return infiniStatus_t Status code of the operation.
 */
__INFINI_C __export infiniStatus_t infiniopCreatePagedAttentionPrefillDescriptor(
    infiniopHandle_t handle,
    infiniopPagedAttentionPrefillDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_cache_desc,
    infiniopTensorDescriptor_t v_cache_desc,
    infiniopTensorDescriptor_t block_tables_desc,
    infiniopTensorDescriptor_t seq_lens_desc,
    infiniopTensorDescriptor_t cum_seq_lens_q_desc,
    infiniopTensorDescriptor_t alibi_slopes_desc,
    float scale);

/**
 * @brief Retrieves the workspace size required for the Paged Attention Prefill operation.
 */
__INFINI_C __export infiniStatus_t infiniopGetPagedAttentionPrefillWorkspaceSize(
    infiniopPagedAttentionPrefillDescriptor_t desc, size_t *size);

/**
 * @brief Executes the Paged Attention Prefill operation.
 * @param desc The Paged Attention Prefill descriptor.
 * @param workspace Pointer to the workspace memory.
 * @param workspace_size The size of the workspace.
 * @param out Pointer to the output tensor data.
 * @param q Pointer to the query tensor data (packed).
 * @param k_cache Pointer to the global key cache data.
 * @param v_cache Pointer to the global value cache data.
 * @param block_tables Pointer to the block tables data.
 * @param seq_lens Pointer to the KV lengths data.
 * @param cum_seq_lens_q Pointer to the Q cumulative sequence lengths data (prefix sum).
 * @param alibi_slopes Pointer to the ALiBi slopes data. Can be NULL.
 * @param stream The device stream (e.g., cudaStream_t) for the operation.
 * @return infiniStatus_t Status code of the operation.
 */
__INFINI_C __export infiniStatus_t infiniopPagedAttentionPrefill(
    infiniopPagedAttentionPrefillDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *q,
    const void *k_cache,
    const void *v_cache,
    const void *block_tables,
    const void *seq_lens,
    const void *cum_seq_lens_q,
    const void *alibi_slopes,
    void *stream);

/**
 * @brief Destroys a Paged Attention Prefill descriptor.
 */
__INFINI_C __export infiniStatus_t infiniopDestroyPagedAttentionPrefillDescriptor(
    infiniopPagedAttentionPrefillDescriptor_t desc);

#endif // __INFINIOP_PAGED_ATTENTION_PREFILL_API_H__
