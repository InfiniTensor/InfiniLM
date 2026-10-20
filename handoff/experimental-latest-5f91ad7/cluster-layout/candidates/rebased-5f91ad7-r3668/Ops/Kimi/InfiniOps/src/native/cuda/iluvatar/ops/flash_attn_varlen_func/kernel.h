#ifndef INFINI_OPS_ILUVATAR_FLASH_ATTN_VARLEN_FUNC_KERNEL_H_
#define INFINI_OPS_ILUVATAR_FLASH_ATTN_VARLEN_FUNC_KERNEL_H_

#include <cassert>
#include <cmath>
#include <optional>
#include <vector>

#include "base/flash_attn_varlen_func.h"
#include "native/cuda/iluvatar/caster.cuh"
#include "native/cuda/iluvatar/runtime_.h"
#include "native/cuda/ops/paged_attention_prefill_infinilm/kernel.h"

namespace infini::ops {
namespace flash_attn_varlen_func_iluvatar_detail {

inline Tensor LegacyCacheView(Tensor cache) {
  return {cache.data(),
          {cache.size(0), cache.size(2), cache.size(1), cache.size(3)},
          cache.dtype(),
          cache.device(),
          {cache.stride(0), cache.stride(2), cache.stride(1), cache.stride(3)}};
}

// The compatibility kernel consumes one total KV length per sequence. This
// metadata view keeps the canonical cumulative buffer and lets the native
// kernel derive adjacent differences without allocating another device tensor.
inline Tensor CumulativeLengthsView(Tensor cumulative_lengths) {
  return {cumulative_lengths.data(),
          {cumulative_lengths.size(0) - 1},
          cumulative_lengths.dtype(),
          cumulative_lengths.device(),
          {cumulative_lengths.stride(0)}};
}

}  // namespace flash_attn_varlen_func_iluvatar_detail

template <>
class Operator<FlashAttnVarlenFunc, Device::Type::kIluvatar>
    : public FlashAttnVarlenFunc {
 public:
  using FlashAttnVarlenFunc::FlashAttnVarlenFunc;
  using FlashAttnVarlenFunc::operator();

  void operator()(const Tensor q, const Tensor k, const Tensor v,
                  const Tensor cu_seqlens_q, const Tensor cu_seqlens_k,
                  const std::optional<Tensor> alibi_slopes,
                  const std::optional<Tensor> block_table,
                  const int64_t /*max_seqlen_q*/,
                  const int64_t /*max_seqlen_k*/, const double dropout_p,
                  const std::optional<double> softmax_scale, const bool causal,
                  const std::vector<int64_t> window_size, const double softcap,
                  const bool deterministic, const bool return_attn_probs,
                  Tensor out, std::optional<Tensor> softmax_lse,
                  std::optional<Tensor> s_dmask) const override {
    assert((q.size(2) == 64 || q.size(2) == 128) &&
           "Iluvatar native `FlashAttnVarlenFunc` supports only head "
           "dimensions 64 and 128.");
    assert(block_table && dropout_p == 0.0 && causal &&
           window_size == std::vector<int64_t>({-1, -1}) && softcap == 0.0 &&
           !deterministic && !return_attn_probs && !softmax_lse && !s_dmask &&
           (!alibi_slopes || alibi_slopes->ndim() == 1) &&
           "Iluvatar native `FlashAttnVarlenFunc` supports only causal paged "
           "inference without dropout, local windows, softcap, deterministic "
           "mode, or auxiliary outputs.");

    auto legacy_k_cache =
        flash_attn_varlen_func_iluvatar_detail::LegacyCacheView(k);
    auto legacy_v_cache =
        flash_attn_varlen_func_iluvatar_detail::LegacyCacheView(v);
    auto cumulative_lengths =
        flash_attn_varlen_func_iluvatar_detail::CumulativeLengthsView(
            cu_seqlens_k);
    const float scale = static_cast<float>(softmax_scale.value_or(
        1.0 / std::sqrt(static_cast<double>(q.size(2)))));

    CudaPagedAttentionPrefillInfinilm<Runtime<Device::Type::kIluvatar>, true>
        provider{q,
                 legacy_k_cache,
                 legacy_v_cache,
                 *block_table,
                 cumulative_lengths,
                 cu_seqlens_q,
                 alibi_slopes,
                 scale,
                 out};
    provider.set_stream(stream_);
    provider(q, legacy_k_cache, legacy_v_cache, *block_table,
             cumulative_lengths, cu_seqlens_q, alibi_slopes, scale, out);
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_ILUVATAR_FLASH_ATTN_VARLEN_FUNC_KERNEL_H_
