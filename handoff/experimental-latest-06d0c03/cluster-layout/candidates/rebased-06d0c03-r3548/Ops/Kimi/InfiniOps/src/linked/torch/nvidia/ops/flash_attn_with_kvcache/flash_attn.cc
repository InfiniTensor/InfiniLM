#include "linked/torch/nvidia/ops/flash_attn_with_kvcache/flash_attn.h"

namespace flash {

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
    int num_splits);

}  // namespace flash

namespace infini::ops::linked::torch::nvidia {

std::vector<at::Tensor> FlashAttnKvcache::Call(
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
  return flash::mha_fwd_kvcache(
      q, k_cache, v_cache, k, v, cache_seqlens, rotary_cos, rotary_sin,
      cache_batch_idx, cache_leftpad, block_table, alibi_slopes, out,
      softmax_scale, causal, window_size_left, window_size_right, softcap,
      rotary_interleaved, num_splits);
}

}  // namespace infini::ops::linked::torch::nvidia

namespace infini::ops::linked::torch {

template class TorchFlashAttnWithKvcache<
    ::infini::ops::linked::torch::nvidia::FlashAttnKvcache>;

}  // namespace infini::ops::linked::torch
