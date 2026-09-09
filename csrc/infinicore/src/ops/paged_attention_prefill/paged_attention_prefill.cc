#include "infinicore/ops/paged_attention_prefill.hpp"
#include "infinicore/ops/mha_varlen.hpp"

#include "../../utils.hpp"

#include <cstdint>
#include <limits>
#include <vector>

namespace infinicore::op {

common::OpDispatcher<PagedAttentionPrefill::schema> &PagedAttentionPrefill::dispatcher() {
    static common::OpDispatcher<PagedAttentionPrefill::schema> dispatcher_;
    return dispatcher_;
};

void PagedAttentionPrefill::execute(Tensor out, Tensor q, Tensor k_cache, Tensor v_cache,
                                    Tensor block_tables, Tensor kv_lens, Tensor cum_seqlens_q,
                                    std::optional<Tensor> alibi_slopes, float scale) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, q, k_cache, v_cache, block_tables, kv_lens, cum_seqlens_q);
    INFINICORE_ASSERT(kv_lens->ndim() == 1 && kv_lens->dtype() == DataType::kInt32);
    INFINICORE_ASSERT(cum_seqlens_q->ndim() == 1
                      && cum_seqlens_q->size(0) == kv_lens->size(0) + 1);

    auto kv_lens_cpu = kv_lens->to(Device{Device::Type::kCpu});
    const auto *kv_lens_data = reinterpret_cast<const int32_t *>(kv_lens_cpu->data());
    std::vector<int32_t> cum_seqlens_k(kv_lens->size(0) + 1, 0);
    int64_t total_kv_len = 0;
    for (size_t i = 0; i < kv_lens->size(0); ++i) {
        INFINICORE_ASSERT(kv_lens_data[i] >= 0);
        total_kv_len += kv_lens_data[i];
        INFINICORE_ASSERT(total_kv_len <= std::numeric_limits<int32_t>::max());
        cum_seqlens_k[i + 1] = static_cast<int32_t>(total_kv_len);
    }

    auto cum_seqlens_k_tensor = Tensor::empty(
        {cum_seqlens_k.size()}, DataType::kInt32, out->device());
    context::memcpyH2D(
        cum_seqlens_k_tensor->data(),
        cum_seqlens_k.data(),
        cum_seqlens_k.size() * sizeof(int32_t),
        false);

    const auto max_seqlen_k = block_tables->size(1) * k_cache->size(1);
    INFINICORE_ASSERT(q->size(0) <= static_cast<size_t>(std::numeric_limits<int>::max()));
    INFINICORE_ASSERT(max_seqlen_k <= static_cast<size_t>(std::numeric_limits<int>::max()));
    mha_varlen_(
        out,
        q,
        k_cache,
        v_cache,
        cum_seqlens_q,
        cum_seqlens_k_tensor,
        block_tables,
        static_cast<int>(q->size(0)),
        static_cast<int>(max_seqlen_k),
        alibi_slopes,
        scale);
}

Tensor paged_attention_prefill(Tensor q, Tensor k_cache, Tensor v_cache,
                               Tensor block_tables, Tensor kv_lens, Tensor cum_seqlens_q,
                               std::optional<Tensor> alibi_slopes, float scale) {

    auto out = Tensor::empty(q->shape(), q->dtype(), q->device());
    paged_attention_prefill_(out, q, k_cache, v_cache, block_tables, kv_lens, cum_seqlens_q, alibi_slopes, scale);
    return out;
}

void paged_attention_prefill_(Tensor out, Tensor q, Tensor k_cache, Tensor v_cache,
                              Tensor block_tables, Tensor kv_lens, Tensor cum_seqlens_q,
                              std::optional<Tensor> alibi_slopes, float scale) {

    PagedAttentionPrefill::execute(out, q, k_cache, v_cache, block_tables, kv_lens, cum_seqlens_q, alibi_slopes, scale);
}

} // namespace infinicore::op
