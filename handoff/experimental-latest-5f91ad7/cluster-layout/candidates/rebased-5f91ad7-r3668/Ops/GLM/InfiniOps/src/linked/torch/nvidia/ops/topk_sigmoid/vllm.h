#ifndef INFINI_OPS_LINKED_TORCH_NVIDIA_OPS_TOPK_SIGMOID_VLLM_H_
#define INFINI_OPS_LINKED_TORCH_NVIDIA_OPS_TOPK_SIGMOID_VLLM_H_

#include <optional>

#include "linked/torch/ops/topk_sigmoid.h"
#include "torch/nvidia/c10.h"

namespace infini::ops::linked::torch::nvidia {

struct VllmTopkSigmoid : C10<Device::Type::kNvidia> {
  static void Validate(bool has_is_padding, double routed_scaling_factor);

  static void Call(at::Tensor topk_weights, at::Tensor topk_indices,
                   at::Tensor token_expert_indices, at::Tensor gating_output,
                   bool renormalize, std::optional<at::Tensor> bias);
};

}  // namespace infini::ops::linked::torch::nvidia

namespace infini::ops::linked::torch {

extern template class TorchTopkSigmoid<
    ::infini::ops::linked::torch::nvidia::VllmTopkSigmoid>;

}  // namespace infini::ops::linked::torch

namespace infini::ops {

template <>
class Operator<TopkSigmoid, Device::Type::kNvidia, 16>
    : public linked::torch::TorchTopkSigmoid<
          linked::torch::nvidia::VllmTopkSigmoid> {
 public:
  using linked::torch::TorchTopkSigmoid<
      linked::torch::nvidia::VllmTopkSigmoid>::TorchTopkSigmoid;

  using linked::torch::TorchTopkSigmoid<
      linked::torch::nvidia::VllmTopkSigmoid>::operator();
};

}  // namespace infini::ops

#endif  // INFINI_OPS_LINKED_TORCH_NVIDIA_OPS_TOPK_SIGMOID_VLLM_H_
