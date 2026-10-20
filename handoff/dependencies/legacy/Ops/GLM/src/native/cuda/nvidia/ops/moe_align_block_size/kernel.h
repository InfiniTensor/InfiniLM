#ifndef INFINI_OPS_NVIDIA_MOE_ALIGN_BLOCK_SIZE_KERNEL_H_
#define INFINI_OPS_NVIDIA_MOE_ALIGN_BLOCK_SIZE_KERNEL_H_

#include <optional>

#include "base/moe_align_block_size.h"

namespace infini::ops {

template <>
class Operator<MoeAlignBlockSize, Device::Type::kNvidia, 0>
    : public MoeAlignBlockSize {
 public:
  Operator(const Tensor topk_ids, const std::optional<Tensor> expert_map,
           const int64_t num_experts, const int64_t block_size,
           Tensor sorted_token_ids, Tensor experts_ids,
           Tensor num_tokens_post_pad);

  using MoeAlignBlockSize::operator();

  bool capture_safe() const override { return true; }

 protected:
  void Run(const Tensor topk_ids, const std::optional<Tensor> maybe_expert_map,
           const int64_t num_experts, const int64_t block_size,
           Tensor sorted_token_ids, Tensor experts_ids,
           Tensor num_tokens_post_pad) const override;

 private:
  int device_index_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_NVIDIA_MOE_ALIGN_BLOCK_SIZE_KERNEL_H_
