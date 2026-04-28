#ifndef __KV_CACHING_INFO_H__
#define __KV_CACHING_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"

namespace op::kv_caching {

class KVCachingInfo {
private:
    KVCachingInfo() = default;

public:
    infiniDtype_t dtype;
    infiniDtype_t past_len_dtype;
    size_t batch_size, num_kv_heads, max_seq_len, seq_len, hidden_dim;
    ptrdiff_t k_cache_strides_0, k_cache_strides_1, k_cache_strides_2, k_cache_strides_3;
    ptrdiff_t v_cache_strides_0, v_cache_strides_1, v_cache_strides_2, v_cache_strides_3;
    ptrdiff_t k_strides_0, k_strides_1, k_strides_2, k_strides_3;
    ptrdiff_t v_strides_0, v_strides_1, v_strides_2, v_strides_3;

    static utils::Result<KVCachingInfo> createKVCachingInfo(
        infiniopTensorDescriptor_t k_cache,
        infiniopTensorDescriptor_t v_cache,
        infiniopTensorDescriptor_t k,
        infiniopTensorDescriptor_t v,
        infiniopTensorDescriptor_t past_kv_lengths) {

        CHECK_OR_RETURN(
            k_cache != nullptr && v_cache != nullptr && k != nullptr && v != nullptr && past_kv_lengths != nullptr,
            INFINI_STATUS_NULL_POINTER);

        const infiniDtype_t dtype = k_cache->dtype();
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);
        const infiniDtype_t past_len_dtype = past_kv_lengths->dtype();
        CHECK_DTYPE(past_len_dtype, INFINI_DTYPE_I32, INFINI_DTYPE_I64);

        CHECK_OR_RETURN(k_cache->ndim() == 4
                            && v_cache->ndim() == 4
                            && k->ndim() == 4
                            && v->ndim() == 4,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);

        auto shape = k_cache->shape();
        CHECK_SAME_SHAPE(shape, v_cache->shape());
        CHECK_SAME_SHAPE(k->shape(), v->shape());

        size_t batch_size = shape[0];
        size_t num_kv_heads = shape[1];
        size_t max_seq_len = shape[2];
        size_t hidden_dim = shape[3];

        size_t seq_len = k->shape()[2];

        CHECK_OR_RETURN(batch_size == k->dim(0)
                            || num_kv_heads == k->dim(1)
                            || hidden_dim == k->dim(3),
                        INFINI_STATUS_BAD_TENSOR_SHAPE);

        ptrdiff_t k_cache_strides_0 = k_cache->strides()[0];
        ptrdiff_t k_cache_strides_1 = k_cache->strides()[1];
        ptrdiff_t k_cache_strides_2 = k_cache->strides()[2];
        ptrdiff_t k_cache_strides_3 = k_cache->strides()[3];

        ptrdiff_t v_cache_strides_0 = v_cache->strides()[0];
        ptrdiff_t v_cache_strides_1 = v_cache->strides()[1];
        ptrdiff_t v_cache_strides_2 = v_cache->strides()[2];
        ptrdiff_t v_cache_strides_3 = v_cache->strides()[3];

        ptrdiff_t k_strides_0 = k->strides()[0];
        ptrdiff_t k_strides_1 = k->strides()[1];
        ptrdiff_t k_strides_2 = k->strides()[2];
        ptrdiff_t k_strides_3 = k->strides()[3];

        ptrdiff_t v_strides_0 = v->strides()[0];
        ptrdiff_t v_strides_1 = v->strides()[1];
        ptrdiff_t v_strides_2 = v->strides()[2];
        ptrdiff_t v_strides_3 = v->strides()[3];

        return utils::Result<KVCachingInfo>(KVCachingInfo{
            dtype,
            past_len_dtype,
            batch_size,
            num_kv_heads,
            max_seq_len,
            seq_len,
            hidden_dim,
            k_cache_strides_0,
            k_cache_strides_1,
            k_cache_strides_2,
            k_cache_strides_3,
            v_cache_strides_0,
            v_cache_strides_1,
            v_cache_strides_2,
            v_cache_strides_3,
            k_strides_0,
            k_strides_1,
            k_strides_2,
            k_strides_3,
            v_strides_0,
            v_strides_1,
            v_strides_2,
            v_strides_3});
    }
};
} // namespace op::kv_caching

#endif //  __KV_CACHING_INFO_H__
