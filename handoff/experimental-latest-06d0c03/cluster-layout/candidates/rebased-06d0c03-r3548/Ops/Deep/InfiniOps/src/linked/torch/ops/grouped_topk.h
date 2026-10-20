#ifndef INFINI_OPS_LINKED_TORCH_OPS_GROUPED_TOPK_H_
#define INFINI_OPS_LINKED_TORCH_OPS_GROUPED_TOPK_H_

#include <utility>

#include "base/grouped_topk.h"
#include "torch/tensor_.h"

namespace infini::ops::linked::torch {

template <typename Backend>
class TorchGroupedTopk : public ::infini::ops::GroupedTopk {
 public:
  using ::infini::ops::GroupedTopk::GroupedTopk;

  using ::infini::ops::GroupedTopk::operator();

  void operator()(const Tensor scores, const Tensor bias,
                  const int64_t num_expert_group, const int64_t topk_group,
                  const int64_t topk, const bool renormalize,
                  const double routed_scaling_factor,
                  const int64_t scoring_func, Tensor topk_values,
                  Tensor topk_indices) const override {
    ValidateCallMetadata(scores, bias, num_expert_group, topk_group, topk,
                         renormalize, routed_scaling_factor, scoring_func,
                         topk_values, topk_indices);

    const typename Backend::StreamGuard stream_guard{
        Backend::GetStreamFromExternal(stream_, device_index_)};
    auto at_scores = ToAtenTensor<Backend::kDeviceType>(
        const_cast<void*>(scores.data()), scores.shape(), scores.strides(),
        scores.dtype(), device_index_);
    auto at_bias = ToAtenTensor<Backend::kDeviceType>(
        const_cast<void*>(bias.data()), bias.shape(), bias.strides(),
        bias.dtype(), device_index_);
    auto at_topk_values = ToAtenTensor<Backend::kDeviceType>(
        topk_values.data(), topk_values.shape(), topk_values.strides(),
        topk_values.dtype(), device_index_);
    auto at_topk_indices = ToAtenTensor<Backend::kDeviceType>(
        topk_indices.data(), topk_indices.shape(), topk_indices.strides(),
        topk_indices.dtype(), device_index_);

    auto [provider_values, provider_indices] = Backend::Call(
        std::move(at_scores), std::move(at_bias), num_expert_group, topk_group,
        topk, renormalize, routed_scaling_factor, scoring_func);
    at_topk_values.copy_(provider_values);
    at_topk_indices.copy_(provider_indices);
  }
};

}  // namespace infini::ops::linked::torch

#endif  // INFINI_OPS_LINKED_TORCH_OPS_GROUPED_TOPK_H_
