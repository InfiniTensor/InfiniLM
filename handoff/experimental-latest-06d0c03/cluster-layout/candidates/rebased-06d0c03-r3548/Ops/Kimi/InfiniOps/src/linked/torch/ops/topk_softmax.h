#ifndef INFINI_OPS_LINKED_TORCH_OPS_TOPK_SOFTMAX_H_
#define INFINI_OPS_LINKED_TORCH_OPS_TOPK_SOFTMAX_H_

#include <c10/util/Exception.h>

#include <optional>
#include <utility>

#include "base/topk_softmax.h"
#include "torch/tensor_.h"

namespace infini::ops::linked::torch {

template <typename Backend>
class TorchTopkSoftmax : public ::infini::ops::TopkSoftmax {
 public:
  using ::infini::ops::TopkSoftmax::TopkSoftmax;
  using ::infini::ops::TopkSoftmax::operator();

  void operator()(const Tensor gating_output, std::optional<Tensor> bias,
                  std::optional<Tensor> is_padding, const bool renormalize,
                  Tensor topk_weights, Tensor topk_indices,
                  Tensor token_expert_indices) const override {
    ValidateCallMetadata(gating_output, bias, is_padding, renormalize,
                         topk_weights, topk_indices, token_expert_indices);
    TORCH_CHECK(!is_padding.has_value(),
                "Linked `topk_softmax` does not support `is_padding`");

    if (num_tokens_ == 0) {
      return;
    }

    const typename Backend::StreamGuard stream_guard{
        Backend::GetStreamFromExternal(stream_, device_index_)};
    auto at_gating_output = ToAtenTensor<Backend::kDeviceType>(
        const_cast<void*>(gating_output.data()), gating_output.shape(),
        gating_output.strides(), gating_output.dtype(), device_index_);
    auto at_topk_weights = ToAtenTensor<Backend::kDeviceType>(
        topk_weights.data(), topk_weights.shape(), topk_weights.strides(),
        topk_weights.dtype(), device_index_);
    auto at_topk_indices = ToAtenTensor<Backend::kDeviceType>(
        topk_indices.data(), topk_indices.shape(), topk_indices.strides(),
        topk_indices.dtype(), device_index_);
    auto at_token_expert_indices = ToAtenTensor<Backend::kDeviceType>(
        token_expert_indices.data(), token_expert_indices.shape(),
        token_expert_indices.strides(), token_expert_indices.dtype(),
        device_index_);

    std::optional<at::Tensor> at_bias;
    if (bias.has_value()) {
      at_bias.emplace(ToAtenTensor<Backend::kDeviceType>(
          const_cast<void*>(bias->data()), bias->shape(), bias->strides(),
          bias->dtype(), device_index_));
    }

    Backend::Call(std::move(at_topk_weights), std::move(at_topk_indices),
                  std::move(at_token_expert_indices),
                  std::move(at_gating_output), renormalize, std::move(at_bias));
  }
};

}  // namespace infini::ops::linked::torch

#endif  // INFINI_OPS_LINKED_TORCH_OPS_TOPK_SOFTMAX_H_
