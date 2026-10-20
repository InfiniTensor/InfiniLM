#include "linked/torch/metax/ops/flash_attn_with_kvcache/flash_attn.h"

#include "linked/torch/ops/flash_attn_with_kvcache.h"
#include "torch/metax/c10.h"

std::vector<at::Tensor> mha_fwd_kvcache(
    at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache,
    std::optional<const at::Tensor>& k, std::optional<const at::Tensor>& v,
    std::optional<const at::Tensor>& cache_seqlens,
    std::optional<const at::Tensor>& rotary_cos,
    std::optional<const at::Tensor>& rotary_sin,
    std::optional<const at::Tensor>& cache_batch_idx,
    std::optional<const at::Tensor>& cache_leftpad,
    std::optional<at::Tensor>& block_table,
    std::optional<at::Tensor>& alibi_slopes, std::optional<at::Tensor>& out,
    float softmax_scale, bool causal, int window_size_left,
    int window_size_right, float softcap, bool rotary_interleaved,
    int num_splits, std::optional<at::Tensor>& flash_attn_mars_ext);

namespace infini::ops::linked::torch::metax {

struct FlashAttnKvcache : C10<Device::Type::kMetax> {
  static std::vector<at::Tensor> Call(
      at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache,
      std::optional<const at::Tensor>& k, std::optional<const at::Tensor>& v,
      std::optional<const at::Tensor>& cache_seqlens,
      std::optional<const at::Tensor>& rotary_cos,
      std::optional<const at::Tensor>& rotary_sin,
      std::optional<const at::Tensor>& cache_batch_idx,
      std::optional<const at::Tensor>& cache_leftpad,
      std::optional<at::Tensor>& block_table,
      std::optional<at::Tensor>& alibi_slopes, std::optional<at::Tensor>& out,
      float softmax_scale, bool causal, int window_size_left,
      int window_size_right, float softcap, bool rotary_interleaved,
      int num_splits) {
    std::optional<at::Tensor> flash_attn_mars_ext;
    return ::mha_fwd_kvcache(
        q, k_cache, v_cache, k, v, cache_seqlens, rotary_cos, rotary_sin,
        cache_batch_idx, cache_leftpad, block_table, alibi_slopes, out,
        softmax_scale, causal, window_size_left, window_size_right, softcap,
        rotary_interleaved, num_splits, flash_attn_mars_ext);
  }
};

}  // namespace infini::ops::linked::torch::metax

namespace infini::ops {

void Operator<FlashAttnWithKvcache, Device::Type::kMetax, 16>::operator()(
    const Tensor q, Tensor k_cache, Tensor v_cache,
    const std::optional<Tensor> k, const std::optional<Tensor> v,
    const std::optional<Tensor> rotary_cos,
    const std::optional<Tensor> rotary_sin, const int64_t cache_seqlens,
    const std::optional<Tensor> cache_batch_idx,
    const std::optional<Tensor> cache_leftpad,
    const std::optional<Tensor> block_table,
    const std::optional<Tensor> alibi_slopes,
    const std::optional<double> softmax_scale, const bool causal,
    const std::vector<int64_t> window_size, const double softcap,
    const bool rotary_interleaved, const int64_t num_splits,
    const bool return_softmax_lse, Tensor out,
    std::optional<Tensor> softmax_lse) const {
  using Delegate = linked::torch::TorchFlashAttnWithKvcache<
      linked::torch::metax::FlashAttnKvcache>;
  if (!delegate_) {
    delegate_ = std::make_unique<Delegate>(
        q, k_cache, v_cache, k, v, rotary_cos, rotary_sin, cache_seqlens,
        cache_batch_idx, cache_leftpad, block_table, alibi_slopes,
        softmax_scale, causal, window_size, softcap, rotary_interleaved,
        num_splits, return_softmax_lse, out, softmax_lse);
  }
  delegate_->set_stream(stream_);
  (*delegate_)(q, k_cache, v_cache, k, v, rotary_cos, rotary_sin, cache_seqlens,
               cache_batch_idx, cache_leftpad, block_table, alibi_slopes,
               softmax_scale, causal, window_size, softcap, rotary_interleaved,
               num_splits, return_softmax_lse, out, softmax_lse);
}

void Operator<FlashAttnWithKvcache, Device::Type::kMetax, 16>::operator()(
    const Tensor q, Tensor k_cache, Tensor v_cache,
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
    std::optional<Tensor> softmax_lse) const {
  using Delegate = linked::torch::TorchFlashAttnWithKvcache<
      linked::torch::metax::FlashAttnKvcache>;
  if (!delegate_) {
    delegate_ = std::make_unique<Delegate>(
        q, k_cache, v_cache, k, v, rotary_cos, rotary_sin, cache_seqlens,
        cache_batch_idx, cache_leftpad, block_table, alibi_slopes,
        softmax_scale, causal, window_size, softcap, rotary_interleaved,
        num_splits, return_softmax_lse, out, softmax_lse);
  }
  delegate_->set_stream(stream_);
  (*delegate_)(q, k_cache, v_cache, k, v, rotary_cos, rotary_sin, cache_seqlens,
               cache_batch_idx, cache_leftpad, block_table, alibi_slopes,
               softmax_scale, causal, window_size, softcap, rotary_interleaved,
               num_splits, return_softmax_lse, out, softmax_lse);
}

}  // namespace infini::ops
