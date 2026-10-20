#ifndef INFINI_OPS_NVIDIA_TOPK_SOFTMAX_KERNEL_H_
#define INFINI_OPS_NVIDIA_TOPK_SOFTMAX_KERNEL_H_

#include <optional>

#include "base/topk_softmax.h"

namespace infini::ops {

template <>
class Operator<TopkSoftmax, Device::Type::kNvidia, 0> : public TopkSoftmax {
 public:
  using TopkSoftmax::TopkSoftmax;

  Operator(const Config& config, const Tensor input, std::optional<Tensor> bias,
           std::optional<Tensor> padding, bool renormalize, Tensor weights,
           Tensor indices, Tensor scratch)
      : TopkSoftmax(input, bias, padding, renormalize, weights, indices,
                    scratch) {
    softcap_ = TopkSoftmaxConfig::SoftcapFrom(config);
    min_sum_ = TopkSoftmaxConfig::MinSumFrom(config);
  }

  static bool supports(const Config&, const Tensor, std::optional<Tensor>,
                       std::optional<Tensor>, bool, Tensor, Tensor, Tensor) {
    return true;
  }

  bool capture_safe() const override { return true; }

  void operator()(const Tensor gating_output, std::optional<Tensor> bias,
                  std::optional<Tensor> is_padding, const bool renormalize,
                  Tensor topk_weights, Tensor topk_indices,
                  Tensor token_expert_indices) const override;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_NVIDIA_TOPK_SOFTMAX_KERNEL_H_
