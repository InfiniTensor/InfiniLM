#ifndef INFINI_OPS_MOORE_FLASH_ATTN_WITH_KVCACHE_PAGED_H_
#define INFINI_OPS_MOORE_FLASH_ATTN_WITH_KVCACHE_PAGED_H_

#include "base/flash_attn_with_kvcache.h"

namespace infini::ops {

template <>
class Operator<FlashAttnWithKvcache, Device::Type::kMoore, 8>
    : public FlashAttnWithKvcache {
 public:
  using FlashAttnWithKvcache::FlashAttnWithKvcache;
  using FlashAttnWithKvcache::operator();

  void operator()(const Tensor q, Tensor k_cache, Tensor v_cache,
                  const std::optional<Tensor> k, const std::optional<Tensor> v,
                  const std::optional<Tensor> rotary_cos,
                  const std::optional<Tensor> rotary_sin,
                  const int64_t cache_seqlens,
                  const std::optional<Tensor> cache_batch_idx,
                  const std::optional<Tensor> cache_leftpad,
                  const std::optional<Tensor> block_table,
                  const std::optional<Tensor> alibi_slopes,
                  const std::optional<double> softmax_scale, const bool causal,
                  const std::vector<int64_t> window_size, const double softcap,
                  const bool rotary_interleaved, const int64_t num_splits,
                  const bool return_softmax_lse, Tensor out,
                  std::optional<Tensor> softmax_lse) const override;

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
                  const bool rotary_interleaved, const int64_t num_splits,
                  const bool return_softmax_lse, Tensor out,
                  std::optional<Tensor> softmax_lse) const override;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_MOORE_FLASH_ATTN_WITH_KVCACHE_PAGED_H_
