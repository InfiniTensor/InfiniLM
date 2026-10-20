#ifndef INFINI_OPS_LINKED_TORCH_HYGON_OPS_FLASH_ATTN_WITH_KVCACHE_FLASH_ATTN_H_
#define INFINI_OPS_LINKED_TORCH_HYGON_OPS_FLASH_ATTN_WITH_KVCACHE_FLASH_ATTN_H_

#include <memory>
#include <cuda_runtime_api.h>
#include <algorithm>

#include "base/flash_attn_with_kvcache.h"

namespace infini::ops {

template <>
class Operator<FlashAttnWithKvcache, Device::Type::kHygon, 16>
    : public FlashAttnWithKvcache {
 public:
  using FlashAttnWithKvcache::FlashAttnWithKvcache;
  using FlashAttnWithKvcache::operator();
  using FlashAttnWithKvcache::supports;

  // The explicit logical bound is enforced before launching the linked kernel.
  // Metadata-only supports remains side-effect free; GPU lengths are checked
  // at invocation. This eager baseline deliberately remains capture-unsafe.
  template<class Lengths>
  static bool supports(const Config &config, const Tensor q, Tensor k_cache, Tensor v_cache,
      const std::optional<Tensor> k, const std::optional<Tensor> v,
      const std::optional<Tensor> rotary_cos, const std::optional<Tensor> rotary_sin,
      const Lengths cache_seqlens, const std::optional<Tensor> cache_batch_idx,
      const std::optional<Tensor> cache_leftpad, const std::optional<Tensor> block_table,
      const std::optional<Tensor> alibi_slopes, const std::optional<double> softmax_scale,
      const bool causal, const std::vector<int64_t> window_size, const double softcap,
      const bool rotary_interleaved, const int64_t num_splits,
      const bool return_softmax_lse, Tensor out, std::optional<Tensor> softmax_lse) {
    if (!FlashAttnWithKvcache::supports(q,k_cache,v_cache,k,v,rotary_cos,rotary_sin,cache_seqlens,cache_batch_idx,cache_leftpad,block_table,alibi_slopes,softmax_scale,causal,window_size,softcap,rotary_interleaved,num_splits,return_softmax_lse,out,softmax_lse)) return false;
    const auto bound=FlashAttnWithKvcacheConfig::BoundFrom(config);
    if (!bound) return true;
    if constexpr (std::is_convertible_v<Lengths,int64_t>) return false;
    else return cache_seqlens.has_value() && block_table.has_value()
        && q.size(1)==1 && !k && !v && !rotary_cos && !rotary_sin
        && !cache_batch_idx && !cache_leftpad && !alibi_slopes
        && softcap==0.0 && causal && window_size==std::vector<int64_t>({-1,-1})
        && num_splits==0 && !return_softmax_lse && !softmax_lse
        && (bound-1)/k_cache.size(1)+1 <= static_cast<int64_t>(block_table->size(1));
  }
  template<class Lengths>
  Operator(const Config &config, const Tensor q, Tensor k_cache, Tensor v_cache,
      const std::optional<Tensor> k, const std::optional<Tensor> v,
      const std::optional<Tensor> rotary_cos, const std::optional<Tensor> rotary_sin,
      const Lengths cache_seqlens, const std::optional<Tensor> cache_batch_idx,
      const std::optional<Tensor> cache_leftpad, const std::optional<Tensor> block_table,
      const std::optional<Tensor> alibi_slopes, const std::optional<double> softmax_scale,
      const bool causal, const std::vector<int64_t> window_size, const double softcap,
      const bool rotary_interleaved, const int64_t num_splits,
      const bool return_softmax_lse, Tensor out, std::optional<Tensor> softmax_lse)
      : FlashAttnWithKvcache(q,k_cache,v_cache,k,v,rotary_cos,rotary_sin,cache_seqlens,cache_batch_idx,cache_leftpad,block_table,alibi_slopes,softmax_scale,causal,window_size,softcap,rotary_interleaved,num_splits,return_softmax_lse,out,softmax_lse),
        logical_bound_(FlashAttnWithKvcacheConfig::BoundFrom(config)) {
    if (!supports(config,q,k_cache,v_cache,k,v,rotary_cos,rotary_sin,cache_seqlens,cache_batch_idx,cache_leftpad,block_table,alibi_slopes,softmax_scale,causal,window_size,softcap,rotary_interleaved,num_splits,return_softmax_lse,out,softmax_lse))
      throw std::invalid_argument("Hygon bounded KV-cache arguments unsupported");
  }


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

 private:
  int64_t logical_bound_{0};
  void ValidateLogicalBound(const std::optional<Tensor> &lengths,
                            const std::optional<Tensor> &blocks,
                            const Tensor &cache) const {
    if (!logical_bound_) return;
    if (!lengths || !blocks || !lengths->data() || !blocks->data())
      throw std::invalid_argument("bounded Hygon attention metadata storage missing");
    if (cudaStreamSynchronize(static_cast<cudaStream_t>(stream_))!=cudaSuccess)
      throw std::runtime_error("bounded Hygon attention input drain failed");
    std::vector<int32_t> host_lengths(lengths->numel()),host_blocks(blocks->numel());
    if (cudaMemcpy(host_lengths.data(),lengths->data(),host_lengths.size()*sizeof(int32_t),cudaMemcpyDeviceToHost)!=cudaSuccess
        || cudaMemcpy(host_blocks.data(),blocks->data(),host_blocks.size()*sizeof(int32_t),cudaMemcpyDeviceToHost)!=cudaSuccess)
      throw std::runtime_error("bounded Hygon attention metadata copy failed");
    const auto width=blocks->size(1),page_size=cache.size(1);
    for (size_t request=0;request<host_lengths.size();++request) {
      const auto length=host_lengths[request];
      if (length<=0 || length>logical_bound_)
        throw std::invalid_argument("Hygon KV length exceeds explicit logical bound");
      const auto pages=(static_cast<size_t>(length)+page_size-1)/page_size;
      if (pages>width) throw std::invalid_argument("Hygon KV length exceeds page-table capacity");
      for (size_t page=0;page<pages;++page) {
        const auto physical=host_blocks[request*width+page];
        if (physical<0 || static_cast<size_t>(physical)>=cache.size(0))
          throw std::invalid_argument("Hygon KV block ID outside cache");
      }
    }
  }
  mutable std::unique_ptr<FlashAttnWithKvcache> delegate_;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_LINKED_TORCH_HYGON_OPS_FLASH_ATTN_WITH_KVCACHE_FLASH_ATTN_H_
