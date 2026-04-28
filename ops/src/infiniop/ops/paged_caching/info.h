#ifndef __PAGED_CACHING_INFO_H__
#define __PAGED_CACHING_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <optional>
#include <vector>

namespace op::paged_caching {

class PagedCachingInfo {
    PagedCachingInfo() = default;

public:
    // --- Data Type ---
    infiniDtype_t dtype;

    // --- Shape Dimensions ---
    size_t num_tokens;
    size_t num_kv_heads;
    size_t head_size;
    size_t block_size;

    // --- Strides for Memory Layout ---
    ptrdiff_t k_src_stride;
    ptrdiff_t v_src_stride;
    ptrdiff_t k_cache_block_stride;
    ptrdiff_t v_cache_block_stride;
    ptrdiff_t k_cache_head_stride;
    ptrdiff_t v_cache_head_stride;
    ptrdiff_t k_cache_slot_stride;
    ptrdiff_t v_cache_slot_stride;

    static utils::Result<PagedCachingInfo> create(
        infiniopTensorDescriptor_t k_cache_desc,
        infiniopTensorDescriptor_t v_cache_desc,
        infiniopTensorDescriptor_t k_desc,
        infiniopTensorDescriptor_t v_desc,
        infiniopTensorDescriptor_t slot_mapping_desc) {

        auto dtype = k_desc->dtype();
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);
        if (v_desc->dtype() != dtype || k_cache_desc->dtype() != dtype || v_cache_desc->dtype() != dtype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (slot_mapping_desc->dtype() != INFINI_DTYPE_I64) {
            printf("slot_mapping must be int64_t.\n");
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        if (k_desc->ndim() != 3 || v_desc->ndim() != 3 || k_cache_desc->ndim() < 4 || v_cache_desc->ndim() < 4 || slot_mapping_desc->ndim() != 1) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        // PagedCachingInfo info;
        // --- Extract shape dimensions ---
        auto k_shape = k_desc->shape();
        auto k_cache_shape = k_cache_desc->shape();

        size_t num_tokens = slot_mapping_desc->shape()[0];
        size_t num_kv_heads = k_shape[1];
        size_t head_size = k_shape[2];
        size_t block_size = k_cache_shape[2]; // Assuming [num_blocks, num_heads, block_size, head_size] layout

        // --- Extract strides for memory access ---
        ptrdiff_t k_src_stride = k_desc->stride(0);
        ptrdiff_t v_src_stride = v_desc->stride(0);
        ptrdiff_t k_cache_block_stride = k_cache_desc->stride(0);
        ptrdiff_t v_cache_block_stride = v_cache_desc->stride(0);
        ptrdiff_t k_cache_head_stride = k_cache_desc->stride(1);
        ptrdiff_t v_cache_head_stride = v_cache_desc->stride(1);
        ptrdiff_t k_cache_slot_stride = k_cache_desc->stride(2);
        ptrdiff_t v_cache_slot_stride = v_cache_desc->stride(2);

        return utils::Result<PagedCachingInfo>(PagedCachingInfo{
            dtype,
            num_tokens,
            num_kv_heads,
            head_size,
            block_size,
            k_src_stride,
            v_src_stride,
            k_cache_block_stride,
            v_cache_block_stride,
            k_cache_head_stride,
            v_cache_head_stride,
            k_cache_slot_stride,
            v_cache_slot_stride});
    }
};

} // namespace op::paged_caching

#endif // __PAGED_CACHING_INFO_H__
