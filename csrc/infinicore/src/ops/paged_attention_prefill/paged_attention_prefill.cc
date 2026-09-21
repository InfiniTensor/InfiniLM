#include "infinicore/ops/paged_attention_prefill.hpp"
#include "infinicore/ops/mha_varlen.hpp"

#include "../../utils.hpp"

#include <algorithm>
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
    INFINICORE_ASSERT(out && q && k_cache && v_cache && block_tables && kv_lens && cum_seqlens_q);
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, q, k_cache, v_cache, block_tables, kv_lens, cum_seqlens_q);
    INFINICORE_ASSERT(q->ndim() == 3 && out->ndim() == 3);
    INFINICORE_ASSERT(k_cache->ndim() == 4 && v_cache->ndim() == 4);
    INFINICORE_ASSERT(q->dtype() == k_cache->dtype() && q->dtype() == v_cache->dtype()
                      && q->dtype() == out->dtype());
    INFINICORE_ASSERT(k_cache->size(0) == v_cache->size(0)
                      && k_cache->size(1) == v_cache->size(1)
                      && k_cache->size(2) == v_cache->size(2));
    INFINICORE_ASSERT(k_cache->size(1) > 0 && k_cache->size(2) > 0);
    INFINICORE_ASSERT(q->size(1) > 0 && q->size(1) % k_cache->size(2) == 0);
    INFINICORE_ASSERT(q->size(2) > 0 && q->size(2) == k_cache->size(3));
    INFINICORE_ASSERT(v_cache->size(3) > 0);
    INFINICORE_ASSERT(out->size(0) == q->size(0) && out->size(1) == q->size(1)
                      && out->size(2) == v_cache->size(3));
    INFINICORE_ASSERT(kv_lens->ndim() == 1 && kv_lens->dtype() == DataType::kInt32);
    INFINICORE_ASSERT(kv_lens->is_contiguous());
    INFINICORE_ASSERT(kv_lens->size(0) < static_cast<size_t>(std::numeric_limits<int32_t>::max()));
    INFINICORE_ASSERT(cum_seqlens_q->ndim() == 1
                      && cum_seqlens_q->dtype() == DataType::kInt32
                      && cum_seqlens_q->is_contiguous()
                      && cum_seqlens_q->size(0) == kv_lens->size(0) + 1);
    INFINICORE_ASSERT(block_tables->ndim() == 2 && block_tables->dtype() == DataType::kInt32
                      && block_tables->size(0) == kv_lens->size(0));
    INFINICORE_ASSERT(q->size(0) <= static_cast<size_t>(std::numeric_limits<int32_t>::max()));

    auto kv_lens_cpu = kv_lens->to(Device{Device::Type::kCpu});
    auto cum_seqlens_q_cpu = cum_seqlens_q->to(Device{Device::Type::kCpu});
    const auto *kv_lens_data = reinterpret_cast<const int32_t *>(kv_lens_cpu->data());
    const auto *cum_seqlens_q_data = reinterpret_cast<const int32_t *>(cum_seqlens_q_cpu->data());
    INFINICORE_ASSERT(cum_seqlens_q_data[0] == 0);
    std::vector<int32_t> cum_seqlens_k(kv_lens->size(0) + 1, 0);
    int64_t total_kv_len = 0;
    int32_t max_seqlen_q = 0;
    int32_t max_seqlen_k = 0;
    for (size_t i = 0; i < kv_lens->size(0); ++i) {
        INFINICORE_ASSERT(cum_seqlens_q_data[i + 1] >= cum_seqlens_q_data[i]);
        INFINICORE_ASSERT(static_cast<size_t>(cum_seqlens_q_data[i + 1]) <= q->size(0));
        const auto query_length = cum_seqlens_q_data[i + 1] - cum_seqlens_q_data[i];
        max_seqlen_q = std::max(max_seqlen_q, query_length);

        INFINICORE_ASSERT(kv_lens_data[i] >= 0);
        const auto kv_length = static_cast<size_t>(kv_lens_data[i]);
        const auto block_size = k_cache->size(1);
        const auto blocks_needed = kv_length / block_size + (kv_length % block_size != 0);
        INFINICORE_ASSERT(blocks_needed <= block_tables->size(1));
        max_seqlen_k = std::max(max_seqlen_k, kv_lens_data[i]);
        total_kv_len += kv_lens_data[i];
        INFINICORE_ASSERT(total_kv_len <= std::numeric_limits<int32_t>::max());
        cum_seqlens_k[i + 1] = static_cast<int32_t>(total_kv_len);
    }
    INFINICORE_ASSERT(static_cast<size_t>(cum_seqlens_q_data[kv_lens->size(0)]) == q->size(0));

    auto cum_seqlens_k_tensor = Tensor::empty(
        {cum_seqlens_k.size()}, DataType::kInt32, out->device());
    context::memcpyH2D(
        cum_seqlens_k_tensor->data(),
        cum_seqlens_k.data(),
        cum_seqlens_k.size() * sizeof(int32_t),
        false);

    mha_varlen_(
        out,
        q,
        k_cache,
        v_cache,
        cum_seqlens_q,
        cum_seqlens_k_tensor,
        block_tables,
        static_cast<int>(max_seqlen_q),
        static_cast<int>(max_seqlen_k),
        alibi_slopes,
        scale);
}

Tensor paged_attention_prefill(Tensor q, Tensor k_cache, Tensor v_cache,
                               Tensor block_tables, Tensor kv_lens, Tensor cum_seqlens_q,
                               std::optional<Tensor> alibi_slopes, float scale) {

    INFINICORE_ASSERT(q && v_cache);
    INFINICORE_ASSERT(q->ndim() == 3 && v_cache->ndim() == 4);
    auto shape = q->shape();
    shape.back() = v_cache->size(3);
    auto out = Tensor::empty(shape, q->dtype(), q->device());
    paged_attention_prefill_(out, q, k_cache, v_cache, block_tables, kv_lens, cum_seqlens_q, alibi_slopes, scale);
    return out;
}

void paged_attention_prefill_(Tensor out, Tensor q, Tensor k_cache, Tensor v_cache,
                              Tensor block_tables, Tensor kv_lens, Tensor cum_seqlens_q,
                              std::optional<Tensor> alibi_slopes, float scale) {

    PagedAttentionPrefill::execute(out, q, k_cache, v_cache, block_tables, kv_lens, cum_seqlens_q, alibi_slopes, scale);
}

} // namespace infinicore::op
