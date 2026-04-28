#ifndef __INFINIOP_PAGED_ATTENTION_API_H__
#define __INFINIOP_PAGED_ATTENTION_API_H__

#include "../operator_descriptor.h"

// Define an opaque handle for the Paged Attention descriptor.
typedef struct InfiniopDescriptor *infiniopPagedAttentionDescriptor_t;

/**
 * @brief Creates a descriptor for the Paged Attention v1 operation.
 *
 * @param handle    The library context handle.
 * @param desc_ptr  Pointer to the created descriptor.
 * @param out_desc  [Output] Shape: (num_seqs, num_heads, head_size).
 * The output tensor for the attention mechanism.
 * @param q_desc    [Input]  Shape: (num_seqs, num_heads, head_size).
 * The query tensor.
 * @param k_cache_desc [Input] Shape: (num_blocks, num_kv_heads, block_size, head_size).
 * Paged key cache storing keys for all sequences.
 * @param v_cache_desc [Input] Shape: (num_blocks, num_kv_heads, block_size, head_size).
 * Paged value cache storing values for all sequences.
 * @param block_tables_desc [Input] Shape: (num_seqs, max_num_blocks_per_seq).
 * Maps each sequence to its physical block indices in the cache.
 * Expected DType: int64_t (I64).
 * @param seq_lens_desc [Input] Shape: (num_seqs,).
 * The current logical length of each sequence.
 * Expected DType: int64_t (I64).
 * @param alibi_slopes_desc [Optional] Shape: (num_heads,).
 * Slopes for ALiBi (Attention with Linear Biases). Can be NULL.
 * @param scale     The attention scaling factor (typically 1/sqrt(head_size)).
 * @return infiniStatus_t Status code.
 */
__INFINI_C __export infiniStatus_t infiniopCreatePagedAttentionDescriptor(
    infiniopHandle_t handle,
    infiniopPagedAttentionDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_cache_desc,
    infiniopTensorDescriptor_t v_cache_desc,
    infiniopTensorDescriptor_t block_tables_desc,
    infiniopTensorDescriptor_t seq_lens_desc,
    infiniopTensorDescriptor_t alibi_slopes_desc,
    float scale);

/**
 * @brief Retrieves the workspace size required for the Paged Attention operation.
 *
 * @param desc The Paged Attention descriptor.
 * @param size A pointer to store the required workspace size in bytes.
 * @return infiniStatus_t Status code of the operation.
 */
__INFINI_C __export infiniStatus_t infiniopGetPagedAttentionWorkspaceSize(
    infiniopPagedAttentionDescriptor_t desc, size_t *size);

/**
 * @brief Executes the Paged Attention v1 operation.
 *
 * @param desc The Paged Attention descriptor.
 * @param workspace Pointer to the workspace memory.
 * @param workspace_size The size of the workspace.
 * @param out Pointer to the output tensor data.
 * @param q Pointer to the query tensor data.
 * @param k_cache Pointer to the key cache data.
 * @param v_cache Pointer to the value cache data.
 * @param block_tables Pointer to the block tables data.
 * @param seq_lens Pointer to the sequence lengths data.
 * @param alibi_slopes Pointer to the ALiBi slopes data. Can be NULL.
 * @param stream The CUDA stream for the operation. Can be NULL.
 * @return infiniStatus_t Status code of the operation.
 */
__INFINI_C __export infiniStatus_t infiniopPagedAttention(
    infiniopPagedAttentionDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *q,
    const void *k_cache,
    const void *v_cache,
    const void *block_tables,
    const void *seq_lens,
    const void *alibi_slopes,
    void *stream);

/**
 * @brief Destroys a Paged Attention descriptor.
 *
 * @param desc The descriptor to be destroyed.
 * @return infiniStatus_t Status code of the operation.
 */
__INFINI_C __export infiniStatus_t infiniopDestroyPagedAttentionDescriptor(
    infiniopPagedAttentionDescriptor_t desc);

#endif // __INFINIOP_PAGED_ATTENTION_API_H__
