#ifndef __INFINIOP_PAGED_ATTENTION_PREFILL_INFO_H__
#define __INFINIOP_PAGED_ATTENTION_PREFILL_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <cstring>
#include <iostream>
#include <optional>
#include <vector>

namespace op::paged_attention_prefill {

class PagedAttentionPrefillInfo {
    PagedAttentionPrefillInfo() = default;

public:
    infiniDtype_t dtype;
    infiniDtype_t index_dtype;
    float scale;

    size_t num_seqs;
    size_t total_q_tokens;
    size_t num_heads;
    size_t num_kv_heads;
    size_t head_size;
    size_t page_block_size;
    size_t max_num_blocks_per_seq;
    size_t num_blocks;

    ptrdiff_t q_stride;
    ptrdiff_t q_head_stride;
    ptrdiff_t k_batch_stride;
    ptrdiff_t k_row_stride;
    ptrdiff_t k_head_stride;
    ptrdiff_t v_batch_stride;
    ptrdiff_t v_row_stride;
    ptrdiff_t v_head_stride;
    ptrdiff_t o_stride;
    ptrdiff_t o_head_stride;

    ptrdiff_t block_table_batch_stride;

    static utils::Result<PagedAttentionPrefillInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t q_desc,
        infiniopTensorDescriptor_t k_cache_desc,
        infiniopTensorDescriptor_t v_cache_desc,
        infiniopTensorDescriptor_t block_tables_desc,
        infiniopTensorDescriptor_t total_kv_lens_desc,
        infiniopTensorDescriptor_t cum_seqlens_q_desc,
        const std::optional<infiniopTensorDescriptor_t> &alibi_slopes_desc,
        float scale) {

        auto dtype = q_desc->dtype();
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16);
        if (out_desc->dtype() != dtype || k_cache_desc->dtype() != dtype || v_cache_desc->dtype() != dtype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        // q/out: [total_q, heads, head_dim]
        if (q_desc->ndim() != 3 || out_desc->ndim() != 3) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        // FA2 paged KV layout: [num_blocks, page_block_size, kv_heads, head_dim]
        if (k_cache_desc->ndim() != 4 || v_cache_desc->ndim() != 4) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (block_tables_desc->ndim() != 2 || total_kv_lens_desc->ndim() != 1 || cum_seqlens_q_desc->ndim() != 1) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        CHECK_OR_RETURN(q_desc->stride(2) == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);
        CHECK_OR_RETURN(out_desc->stride(2) == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);
        CHECK_OR_RETURN(k_cache_desc->stride(3) == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);
        CHECK_OR_RETURN(v_cache_desc->stride(3) == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);

        // Index dtypes: allow I32/I64/U32 (v0.4 roadmap allows internal conversion to I32).
        const auto block_tables_dt = block_tables_desc->dtype();
        if (!((block_tables_dt == INFINI_DTYPE_I64) || (block_tables_dt == INFINI_DTYPE_I32) || (block_tables_dt == INFINI_DTYPE_U32))) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        // Index tensors use int32_t to match mainstream paged-attention implementations
        // (e.g., vLLM / FlashAttention2). 32-bit indices needed, but now we also support int64_t.
        if (!((total_kv_lens_desc->dtype() == INFINI_DTYPE_I64) || (total_kv_lens_desc->dtype() == INFINI_DTYPE_I32) || (total_kv_lens_desc->dtype() == INFINI_DTYPE_U32))) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        if (!((cum_seqlens_q_desc->dtype() == INFINI_DTYPE_I64) || (cum_seqlens_q_desc->dtype() == INFINI_DTYPE_I32) || (cum_seqlens_q_desc->dtype() == INFINI_DTYPE_U32))) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        CHECK_OR_RETURN(block_tables_desc->stride(1) == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);

        if (alibi_slopes_desc.has_value() && alibi_slopes_desc.value() != nullptr) {
            if (alibi_slopes_desc.value()->dtype() != INFINI_DTYPE_F32) {
                return INFINI_STATUS_BAD_TENSOR_DTYPE;
            }
            if (alibi_slopes_desc.value()->ndim() != 1) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
            CHECK_OR_RETURN(alibi_slopes_desc.value()->stride(0) == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);
        }

        const auto q_shape = q_desc->shape();
        const auto k_shape = k_cache_desc->shape();

        const size_t total_q_tokens = q_shape[0];
        const size_t num_heads = q_shape[1];
        const size_t head_size = q_shape[2];

        const size_t num_blocks = k_shape[0];
        const size_t page_block_size = k_shape[2];
        const size_t num_kv_heads = k_shape[1];

        if (head_size != 64 && head_size != 128) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (num_heads % num_kv_heads != 0) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        // v_cache must match the inferred K layout.
        const auto v_shape = v_cache_desc->shape();
        if (v_shape[0] != num_blocks || v_shape[3] != head_size) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        if (v_shape[1] != num_kv_heads || v_shape[2] != page_block_size) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        if (v_cache_desc->shape()[0] != k_shape[0] || v_cache_desc->shape()[3] != k_shape[3]) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        if (out_desc->shape()[0] != q_shape[0] || out_desc->shape()[1] != q_shape[1] || out_desc->shape()[2] != q_shape[2]) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        const size_t num_seqs = total_kv_lens_desc->shape()[0];
        if (cum_seqlens_q_desc->shape()[0] != num_seqs + 1) {
            return INFINI_STATUS_BAD_PARAM;
        }

        const size_t max_num_blocks_per_seq = block_tables_desc->shape()[1];

        // Strides (in elements)
        const ptrdiff_t q_stride = q_desc->stride(0);
        const ptrdiff_t q_head_stride = q_desc->stride(1);
        const ptrdiff_t o_stride = out_desc->stride(0);
        const ptrdiff_t o_head_stride = out_desc->stride(1);

        const ptrdiff_t k_batch_stride = k_cache_desc->stride(0);
        const ptrdiff_t k_row_stride = k_cache_desc->stride(2);
        const ptrdiff_t k_head_stride = k_cache_desc->stride(1);

        const ptrdiff_t v_batch_stride = v_cache_desc->stride(0);
        const ptrdiff_t v_row_stride = v_cache_desc->stride(2);
        const ptrdiff_t v_head_stride = v_cache_desc->stride(1);

        const ptrdiff_t block_table_batch_stride = block_tables_desc->stride(0);

        if (const char *dbg = std::getenv("INFINIOP_DEBUG_PREFILL_INFO")) {
            static bool printed = false;
            if (!printed && std::strcmp(dbg, "1") == 0) {
                const auto bt_shape = block_tables_desc->shape();
                std::fprintf(stderr,
                             "[infiniop][flash_attention_prefill][info] k_shape=[%zu,%zu,%zu,%zu] k_strides=[%td,%td,%td,%td] (row_stride=%td head_stride=%td)\n",
                             static_cast<size_t>(k_shape[0]), static_cast<size_t>(k_shape[1]),
                             static_cast<size_t>(k_shape[2]), static_cast<size_t>(k_shape[3]),
                             k_cache_desc->stride(0), k_cache_desc->stride(1), k_cache_desc->stride(2), k_cache_desc->stride(3),
                             k_row_stride, k_head_stride);
                std::fprintf(stderr,
                             "[infiniop][flash_attention_prefill][info] block_tables shape=[%zu,%zu] strides=[%td,%td]\n",
                             static_cast<size_t>(bt_shape[0]), static_cast<size_t>(bt_shape[1]),
                             block_tables_desc->stride(0), block_tables_desc->stride(1));
                printed = true;
            }
        }

        return utils::Result<PagedAttentionPrefillInfo>(PagedAttentionPrefillInfo{
            dtype,
            block_tables_dt,
            scale,
            num_seqs,
            total_q_tokens,
            num_heads,
            num_kv_heads,
            head_size,
            page_block_size,
            max_num_blocks_per_seq,
            num_blocks,
            q_stride,
            q_head_stride,
            k_batch_stride,
            k_row_stride,
            k_head_stride,
            v_batch_stride,
            v_row_stride,
            v_head_stride,
            o_stride,
            o_head_stride,
            block_table_batch_stride,
        });
    }
};
} // namespace op::paged_attention_prefill

#endif
