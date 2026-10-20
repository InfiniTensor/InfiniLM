#ifndef INFINI_OPS_LINKED_TORCH_NVIDIA_OPS_FLASH_ATTN_WITH_KVCACHE_FLASH_ATTN_H_
#define INFINI_OPS_LINKED_TORCH_NVIDIA_OPS_FLASH_ATTN_WITH_KVCACHE_FLASH_ATTN_H_

#include "linked/torch/ops/flash_attn_with_kvcache.h"
#include "torch/nvidia/c10.h"

namespace infini::ops::linked::torch::nvidia {

struct FlashAttnKvcache : C10<Device::Type::kNvidia> {
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
      int num_splits);
};

}  // namespace infini::ops::linked::torch::nvidia

namespace infini::ops::linked::torch {

extern template class TorchFlashAttnWithKvcache<
    ::infini::ops::linked::torch::nvidia::FlashAttnKvcache>;

}  // namespace infini::ops::linked::torch

namespace infini::ops {

template <>
class Operator<FlashAttnWithKvcache, Device::Type::kNvidia, 16>
    : public linked::torch::TorchFlashAttnWithKvcache<
          linked::torch::nvidia::FlashAttnKvcache> {
 public:
  using linked::torch::TorchFlashAttnWithKvcache<
      linked::torch::nvidia::FlashAttnKvcache>::TorchFlashAttnWithKvcache;

  using linked::torch::TorchFlashAttnWithKvcache<
      linked::torch::nvidia::FlashAttnKvcache>::operator();
};

}  // namespace infini::ops

#endif  // INFINI_OPS_LINKED_TORCH_NVIDIA_OPS_FLASH_ATTN_WITH_KVCACHE_FLASH_ATTN_H_
