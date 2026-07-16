#include "infinicore/ops/paged_attention_prefill.hpp"

#ifdef ENABLE_INFINIOPS_API
#include "../infiniops_impl.hpp"

#include "base/paged_attention_prefill_infinilm.h"

#include <optional>

namespace infinicore::op::paged_attention_prefill_impl::infiniops {
namespace {

using TensorMeta = ::infinicore::op::infiniops::TensorMeta;

void calculate(Tensor out,
               Tensor q,
               Tensor k_cache,
               Tensor v_cache,
               Tensor block_tables,
               Tensor kv_lens,
               Tensor cum_seqlens_q,
               std::optional<Tensor> alibi_slopes,
               float scale) {
    INFINICORE_ASSERT(::infinicore::op::infiniops::isSupportedDevice(out->device().type()));
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, q, k_cache, v_cache, block_tables, kv_lens, cum_seqlens_q);
    if (alibi_slopes) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, *alibi_slopes);
    }

    infini::ops::Handle handle;
    handle.set_stream(context::getStream());
    infini::ops::Config config;

    TensorMeta out_meta(out);
    TensorMeta q_meta(q);
    TensorMeta k_cache_meta(k_cache);
    TensorMeta v_cache_meta(v_cache);
    TensorMeta block_tables_meta(block_tables);
    TensorMeta kv_lens_meta(kv_lens);
    TensorMeta cum_seqlens_q_meta(cum_seqlens_q);
    std::optional<TensorMeta> alibi_slopes_meta;
    if (alibi_slopes) {
        alibi_slopes_meta.emplace(*alibi_slopes);
    }

    infini::ops::PagedAttentionPrefillInfinilm::Call(
        handle,
        config,
        q_meta.tensor(q),
        k_cache_meta.tensor(k_cache),
        v_cache_meta.tensor(v_cache),
        block_tables_meta.tensor(block_tables),
        kv_lens_meta.tensor(kv_lens),
        cum_seqlens_q_meta.tensor(cum_seqlens_q),
        alibi_slopes_meta ? std::optional<infini::ops::Tensor>{alibi_slopes_meta->tensor(*alibi_slopes)} : std::nullopt,
        scale,
        out_meta.tensor(out));
}

} // namespace

static bool registered = []() {
    ::infinicore::op::infiniops::registerSupportedDevices(PagedAttentionPrefill::dispatcher(), &calculate);
    return true;
}();

} // namespace infinicore::op::paged_attention_prefill_impl::infiniops
#endif
