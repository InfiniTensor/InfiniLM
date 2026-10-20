#ifndef INFINI_OPS_LINKED_TORCH_OPS_TOPK_SIGMOID_H_
#define INFINI_OPS_LINKED_TORCH_OPS_TOPK_SIGMOID_H_

#include <optional>
#include <utility>

#include "base/topk_sigmoid.h"
#include "torch/tensor_.h"

namespace infini::ops::linked::torch {

template <typename Backend>
class TorchTopkSigmoid : public ::infini::ops::TopkSigmoid {
 public:
  using ::infini::ops::TopkSigmoid::TopkSigmoid;

  using ::infini::ops::TopkSigmoid::operator();

  void operator()(const Tensor gating_output,
                  std::optional<Tensor> e_score_correction_bias,
                  std::optional<Tensor> is_padding, const bool renormalize,
                  const double routed_scaling_factor, Tensor topk_weights,
                  Tensor topk_ids, Tensor token_expert_indices) const override {
    ValidateCallMetadata(gating_output, e_score_correction_bias, is_padding,
                         renormalize, routed_scaling_factor, topk_weights,
                         topk_ids, token_expert_indices);
    Backend::Validate(is_padding.has_value(), routed_scaling_factor);
    if (num_tokens_ == 0) {
      return;
    }

    const typename Backend::StreamGuard stream_guard{
        Backend::GetStreamFromExternal(stream_, device_index_)};
    auto at_gating_output = ToAtenTensor<Backend::kDeviceType>(
        const_cast<void*>(gating_output.data()), gating_output.shape(),
        gating_output.strides(), gating_output.dtype(), device_index_);
    std::optional<at::Tensor> at_e_score_correction_bias;
    if (e_score_correction_bias.has_value()) {
      at_e_score_correction_bias.emplace(ToAtenTensor<Backend::kDeviceType>(
          const_cast<void*>(e_score_correction_bias->data()),
          e_score_correction_bias->shape(), e_score_correction_bias->strides(),
          e_score_correction_bias->dtype(), device_index_));
    }
    auto at_topk_weights = ToAtenTensor<Backend::kDeviceType>(
        topk_weights.data(), topk_weights.shape(), topk_weights.strides(),
        topk_weights.dtype(), device_index_);
    auto at_topk_ids = ToAtenTensor<Backend::kDeviceType>(
        topk_ids.data(), topk_ids.shape(), topk_ids.strides(), topk_ids.dtype(),
        device_index_);
    auto at_token_expert_indices = ToAtenTensor<Backend::kDeviceType>(
        token_expert_indices.data(), token_expert_indices.shape(),
        token_expert_indices.strides(), token_expert_indices.dtype(),
        device_index_);

    Backend::Call(std::move(at_topk_weights), std::move(at_topk_ids),
                  std::move(at_token_expert_indices),
                  std::move(at_gating_output), renormalize,
                  std::move(at_e_score_correction_bias));
  }
};

}  // namespace infini::ops::linked::torch

#endif  // INFINI_OPS_LINKED_TORCH_OPS_TOPK_SIGMOID_H_
