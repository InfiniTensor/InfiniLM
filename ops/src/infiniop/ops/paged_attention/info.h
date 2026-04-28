#ifndef __PAGED_ATTENTION_INFO_H__
#define __PAGED_ATTENTION_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <iostream>
#include <optional>
#include <vector>

namespace op::paged_attention {

class PagedAttentionInfo {
    PagedAttentionInfo() = default;

public:
    infiniDtype_t dtype;
    infiniDtype_t index_dtype;
    float scale;

    size_t num_seqs;
    size_t num_heads;
    size_t num_kv_heads;
    size_t head_size;
    size_t page_block_size;
    size_t max_num_blocks_per_seq;

    ptrdiff_t q_stride;
    ptrdiff_t k_batch_stride;
    ptrdiff_t k_row_stride;
    ptrdiff_t k_head_stride;
    ptrdiff_t v_batch_stride;
    ptrdiff_t v_row_stride;
    ptrdiff_t v_head_stride;
    ptrdiff_t o_stride;

    ptrdiff_t block_table_batch_stride;
    ptrdiff_t cache_lens_stride;

    static utils::Result<PagedAttentionInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t q_desc,
        infiniopTensorDescriptor_t k_cache_desc,
        infiniopTensorDescriptor_t v_cache_desc,
        infiniopTensorDescriptor_t block_tables_desc,
        infiniopTensorDescriptor_t cache_lens_desc,
        const std::optional<infiniopTensorDescriptor_t> &alibi_slopes_desc,
        float scale) {

        auto dtype = q_desc->dtype();
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16);
        if (out_desc->dtype() != dtype || k_cache_desc->dtype() != dtype || v_cache_desc->dtype() != dtype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        if (q_desc->ndim() != 3 || out_desc->ndim() != 3) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (k_cache_desc->ndim() != 4 || v_cache_desc->ndim() != 4) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (block_tables_desc->ndim() != 2 || cache_lens_desc->ndim() != 1) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        CHECK_OR_RETURN(q_desc->stride(2) == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);
        CHECK_OR_RETURN(out_desc->stride(2) == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);
        CHECK_OR_RETURN(k_cache_desc->stride(3) == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);
        CHECK_OR_RETURN(v_cache_desc->stride(3) == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);

        const auto block_tables_dt = block_tables_desc->dtype();
        const auto cache_lens_dt = cache_lens_desc->dtype();
        const bool debug_dtype = (std::getenv("INFINIOP_FLASH_DEBUG_DTYPE") != nullptr);
        const bool block_tables_ok = (block_tables_dt == INFINI_DTYPE_I64) || (block_tables_dt == INFINI_DTYPE_I32) || (block_tables_dt == INFINI_DTYPE_U32);
        const bool cache_lens_ok = (cache_lens_dt == INFINI_DTYPE_I64) || (cache_lens_dt == INFINI_DTYPE_I32) || (cache_lens_dt == INFINI_DTYPE_U32);
        if (!(block_tables_ok && cache_lens_ok)) {
            if (debug_dtype) {
                std::fprintf(stderr,
                             "[flash_attention] Bad index dtype: block_tables=%d cache_lens=%d (expected I32/I64/U32)\n",
                             static_cast<int>(block_tables_dt), static_cast<int>(cache_lens_dt));
            }
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (block_tables_dt != cache_lens_dt) {
            // Keep them consistent to simplify backend dispatch.
            if (debug_dtype) {
                std::fprintf(stderr,
                             "[flash_attention] Mismatched index dtype: block_tables=%d cache_lens=%d\n",
                             static_cast<int>(block_tables_dt), static_cast<int>(cache_lens_dt));
            }
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        CHECK_OR_RETURN(block_tables_desc->stride(1) == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);
        CHECK_OR_RETURN(cache_lens_desc->stride(0) == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);

        if (alibi_slopes_desc.has_value() && alibi_slopes_desc.value() != nullptr) {
            if (alibi_slopes_desc.value()->dtype() != INFINI_DTYPE_F32) {
                return INFINI_STATUS_BAD_TENSOR_DTYPE;
            }
            if (alibi_slopes_desc.value()->ndim() != 1) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
            CHECK_OR_RETURN(alibi_slopes_desc.value()->stride(0) == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);
        }

        // Shapes
        auto q_shape = q_desc->shape();
        auto k_shape = k_cache_desc->shape();

        const size_t num_seqs = q_shape[0];
        const size_t num_heads = q_shape[1];
        const size_t head_size = q_shape[2];

        const size_t num_blocks = k_shape[0];
        (void)num_blocks;
        const size_t page_block_size = k_shape[2];
        const size_t num_kv_heads = k_shape[1];

        // if (page_block_size % 256 != 0) {
        //     printf("paged block size %zu\n", page_block_size);
        //     return INFINI_STATUS_BAD_TENSOR_SHAPE;
        // }
        if (head_size != 64 && head_size != 128) {
            // First build only targets common FA2 head dims (expand later).
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (num_heads % num_kv_heads != 0) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        if (v_cache_desc->shape()[0] != k_shape[0] || v_cache_desc->shape()[1] != k_shape[1] || v_cache_desc->shape()[2] != k_shape[2] || v_cache_desc->shape()[3] != k_shape[3]) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        if (out_desc->shape()[0] != q_shape[0] || out_desc->shape()[1] != q_shape[1] || out_desc->shape()[2] != q_shape[2]) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        if (cache_lens_desc->shape()[0] != num_seqs) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        const size_t max_num_blocks_per_seq = block_tables_desc->shape()[1];

        // Strides (in elements)
        const ptrdiff_t q_stride = q_desc->stride(0);
        const ptrdiff_t o_stride = out_desc->stride(0);

        const ptrdiff_t k_batch_stride = k_cache_desc->stride(0);
        const ptrdiff_t k_row_stride = k_cache_desc->stride(2);
        const ptrdiff_t k_head_stride = k_cache_desc->stride(1);

        const ptrdiff_t v_batch_stride = v_cache_desc->stride(0);
        const ptrdiff_t v_row_stride = v_cache_desc->stride(2);
        const ptrdiff_t v_head_stride = v_cache_desc->stride(1);

        const ptrdiff_t block_table_batch_stride = block_tables_desc->stride(0);
        const ptrdiff_t cache_lens_stride = cache_lens_desc->stride(0);

        return utils::Result<PagedAttentionInfo>(PagedAttentionInfo{
            dtype,
            block_tables_dt,
            scale,
            num_seqs,
            num_heads,
            num_kv_heads,
            head_size,
            page_block_size,
            max_num_blocks_per_seq,
            q_stride,
            k_batch_stride,
            k_row_stride,
            k_head_stride,
            v_batch_stride,
            v_row_stride,
            v_head_stride,
            o_stride,
            block_table_batch_stride,
            cache_lens_stride,
        });
    }
};

} // namespace op::paged_attention

#endif // __PAGED_ATTENTION_INFO_H__
