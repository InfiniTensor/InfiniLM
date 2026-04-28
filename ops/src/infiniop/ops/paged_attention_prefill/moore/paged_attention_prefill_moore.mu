#include <musa_fp16.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

#include "../../../devices/moore/moore_common.h"
#include "../../../devices/moore/moore_kernel_common.h"
#include "paged_attention_prefill_kernel.h"
#include "paged_attention_prefill_moore.h"

template <typename Tindex, typename Tdata, typename Tcompute>
infiniStatus_t launchPagedAttentionPrefill(
    Tdata *out, const Tdata *q, const Tdata *k_cache, const Tdata *v_cache,
    const Tindex *block_tables,
    const Tindex *seq_lens,
    const Tindex *cum_seq_lens_q,
    const float *alibi_slopes,
    const size_t num_heads,
    const size_t num_seqs,
    const size_t num_kv_heads,
    const float scale,
    const size_t max_num_blocks_per_seq,
    const size_t page_block_size,
    const size_t total_q_tokens,
    const size_t head_size,
    const ptrdiff_t k_batch_stride,
    const ptrdiff_t k_head_stride,
    const ptrdiff_t q_stride,
    const ptrdiff_t q_head_stride,
    musaStream_t stream) {

    if (total_q_tokens == 0 || num_heads == 0) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    dim3 grid(total_q_tokens, num_heads);
    dim3 block(head_size);

    op::paged_attention_prefill::cuda::pagedAttentionPrefillKernel<Tindex, Tdata, Tcompute>
        <<<grid, block, 0, stream>>>(
            out, q, k_cache, v_cache,
            block_tables, seq_lens, cum_seq_lens_q, alibi_slopes,
            num_heads, num_kv_heads, scale,
            max_num_blocks_per_seq, page_block_size,
            k_batch_stride, k_head_stride,
            q_stride, q_head_stride,
            head_size,
            num_seqs);

    return INFINI_STATUS_SUCCESS;
}

namespace op::paged_attention_prefill::moore {

struct Descriptor::Opaque {
    std::shared_ptr<device::moore::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_cache_desc,
    infiniopTensorDescriptor_t v_cache_desc,
    infiniopTensorDescriptor_t block_tables_desc,
    infiniopTensorDescriptor_t seq_lens_desc,
    infiniopTensorDescriptor_t cum_seq_lens_q_desc,
    const std::optional<infiniopTensorDescriptor_t> &alibi_slopes_desc,
    float scale) {

    auto info = PagedAttentionPrefillInfo::create(
        out_desc, q_desc, k_cache_desc, v_cache_desc,
        block_tables_desc, seq_lens_desc,
        cum_seq_lens_q_desc,
        alibi_slopes_desc, scale);

    CHECK_RESULT(info);

    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::moore::Handle *>(handle)->internal()},
        info.take(), 0, handle->device, handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *out, const void *q, const void *k_cache, const void *v_cache,
    const void *block_tables,
    const void *seq_lens,
    const void *cum_seq_lens_q,
    const void *alibi_slopes,
    void *stream_) const {

    musaStream_t stream = (musaStream_t)stream_;

#define DISPATCH_KERNEL(Tindex, Tdata, Tcompute)                                                             \
    return launchPagedAttentionPrefill<Tindex, Tdata, Tcompute>(                                                  \
        (Tdata *)out, (const Tdata *)q, (const Tdata *)k_cache, (const Tdata *)v_cache,            \
        static_cast<const Tindex *>(block_tables), static_cast<const Tindex *>(seq_lens), static_cast<const Tindex *>(cum_seq_lens_q), \
        (const float *)alibi_slopes,                                                               \
        _info.num_heads, _info.num_seqs, _info.num_kv_heads,                                       \
        _info.scale, _info.max_num_blocks_per_seq,                                                 \
        _info.page_block_size, _info.total_q_tokens,                                                    \
        _info.head_size,                                                                           \
        _info.k_batch_stride, _info.k_head_stride,                                               \
        _info.q_stride, _info.q_head_stride,                                                       \
        stream)

#define DISPATCH_INDEX(Tindex)                             \
    do {                                                   \
        if (_info.dtype == INFINI_DTYPE_F16) {             \
            DISPATCH_KERNEL(Tindex, half, float);          \
        }                                                  \
        if (_info.dtype == INFINI_DTYPE_BF16) {            \
            DISPATCH_KERNEL(Tindex, __nv_bfloat16, float); \
        }                                                  \
        return INFINI_STATUS_BAD_TENSOR_DTYPE;             \
    } while (false)

    if (_info.index_dtype == INFINI_DTYPE_I64){
        DISPATCH_INDEX(int64_t);
    } else if (_info.index_dtype == INFINI_DTYPE_I32){
        DISPATCH_INDEX(int32_t);
    } else if (_info.index_dtype == INFINI_DTYPE_U32){
        DISPATCH_INDEX(uint32_t);
    }

    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}

} // namespace op::paged_attention_prefill::moore
