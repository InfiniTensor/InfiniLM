#ifndef INFINI_OPS_NVIDIA_TOPK_SIGMOID_KERNEL_H_
#define INFINI_OPS_NVIDIA_TOPK_SIGMOID_KERNEL_H_

#include <optional>

#include "base/topk_sigmoid.h"

namespace infini::ops {

template <>
class Operator<TopkSigmoid, Device::Type::kNvidia, 0> : public TopkSigmoid {
 public:
  using TopkSigmoid::TopkSigmoid;

  void operator()(const Tensor gating_output,
                  std::optional<Tensor> e_score_correction_bias,
                  std::optional<Tensor> is_padding, const bool renormalize,
                  const double routed_scaling_factor, Tensor topk_weights,
                  Tensor topk_ids, Tensor token_expert_indices) const override;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_NVIDIA_TOPK_SIGMOID_KERNEL_H_
