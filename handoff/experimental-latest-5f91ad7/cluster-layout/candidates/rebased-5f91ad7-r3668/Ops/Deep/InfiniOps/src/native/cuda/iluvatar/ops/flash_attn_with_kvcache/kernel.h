#ifndef INFINI_OPS_ILUVATAR_FLASH_ATTN_WITH_KVCACHE_KERNEL_H_
#define INFINI_OPS_ILUVATAR_FLASH_ATTN_WITH_KVCACHE_KERNEL_H_

#include <cassert>
#include <cmath>
#include <optional>
#include <vector>

#include "base/flash_attn_with_kvcache.h"
#include "native/cuda/iluvatar/caster.cuh"
#include "native/cuda/iluvatar/runtime_.h"
#include "native/cuda/ops/paged_attention_infinilm/kernel.h"

namespace infini::ops {
namespace flash_attn_with_kvcache_iluvatar_detail {

inline Tensor DecodeView(Tensor tensor) {
  return {tensor.data(),
          {tensor.size(0), tensor.size(2), tensor.size(3)},
          tensor.dtype(),
          tensor.device(),
          {tensor.stride(0), tensor.stride(2), tensor.stride(3)}};
}

inline Tensor LegacyCacheView(Tensor cache) {
  return {cache.data(),
          {cache.size(0), cache.size(2), cache.size(1), cache.size(3)},
          cache.dtype(),
          cache.device(),
          {cache.stride(0), cache.stride(2), cache.stride(1), cache.stride(3)}};
}

}  // namespace flash_attn_with_kvcache_iluvatar_detail

template <>
class Operator<FlashAttnWithKvcache, Device::Type::kIluvatar>
    : public FlashAttnWithKvcache {
 public:
  using FlashAttnWithKvcache::FlashAttnWithKvcache;
  using FlashAttnWithKvcache::operator();

  std::size_t workspace_size_in_bytes() const override {
    return static_cast<std::size_t>(kMaxSplits) * q_shape_[0] * q_shape_[2] *
           (head_size_ + 2) * sizeof(float);
  }

  void operator()(const Tensor, Tensor, Tensor, const std::optional<Tensor>,
                  const std::optional<Tensor>, const std::optional<Tensor>,
                  const std::optional<Tensor>, const int64_t,
                  const std::optional<Tensor>, const std::optional<Tensor>,
                  const std::optional<Tensor>, const std::optional<Tensor>,
                  const std::optional<double>, const bool,
                  const std::vector<int64_t>, const double, const bool,
                  const int64_t, const bool, Tensor,
                  std::optional<Tensor>) const override {
    assert(false &&
           "Iluvatar native `FlashAttnWithKvcache` requires tensor "
           "`cache_seqlens`.");
  }

  void operator()(const Tensor q, Tensor k_cache, Tensor v_cache,
                  const std::optional<Tensor> k, const std::optional<Tensor> v,
                  const std::optional<Tensor> rotary_cos,
                  const std::optional<Tensor> rotary_sin,
                  const std::optional<Tensor> cache_seqlens,
                  const std::optional<Tensor> cache_batch_idx,
                  const std::optional<Tensor> cache_leftpad,
                  const std::optional<Tensor> block_table,
                  const std::optional<Tensor> alibi_slopes,
                  const std::optional<double> softmax_scale, const bool causal,
                  const std::vector<int64_t> window_size, const double softcap,
                  const bool /*rotary_interleaved*/,
                  const int64_t /*num_splits*/, const bool return_softmax_lse,
                  Tensor out,
                  std::optional<Tensor> softmax_lse) const override {
    assert((head_size_ == 64 || head_size_ == 128) &&
           "Iluvatar native `FlashAttnWithKvcache` supports only head "
           "dimensions 64 and 128.");
    assert(q.size(1) == 1 && !k && !v && !rotary_cos && !rotary_sin &&
           cache_seqlens && !cache_batch_idx && !cache_leftpad && block_table &&
           causal && window_size == std::vector<int64_t>({-1, -1}) &&
           softcap == 0.0 && !return_softmax_lse && !softmax_lse &&
           (!alibi_slopes || alibi_slopes->ndim() == 1) &&
           "Iluvatar native `FlashAttnWithKvcache` supports only causal paged "
           "decode without KV update, rotary inputs, local windows, softcap, "
           "or auxiliary outputs.");

    auto q_view = flash_attn_with_kvcache_iluvatar_detail::DecodeView(q);
    auto out_view = flash_attn_with_kvcache_iluvatar_detail::DecodeView(out);
    auto legacy_k_cache =
        flash_attn_with_kvcache_iluvatar_detail::LegacyCacheView(k_cache);
    auto legacy_v_cache =
        flash_attn_with_kvcache_iluvatar_detail::LegacyCacheView(v_cache);
    const float scale = static_cast<float>(softmax_scale.value_or(
        1.0 / std::sqrt(static_cast<double>(head_size_))));

    CudaPagedAttentionInfinilm<Runtime<Device::Type::kIluvatar>> provider{
        q_view,         legacy_k_cache, legacy_v_cache, *block_table,
        *cache_seqlens, alibi_slopes,   scale,          out_view};
    provider.set_stream(stream_);
    provider.set_workspace(workspace_);
    provider.set_workspace_size_in_bytes(workspace_size_in_bytes_);
    provider(q_view, legacy_k_cache, legacy_v_cache, *block_table,
             *cache_seqlens, alibi_slopes, scale, out_view);
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_ILUVATAR_FLASH_ATTN_WITH_KVCACHE_KERNEL_H_
