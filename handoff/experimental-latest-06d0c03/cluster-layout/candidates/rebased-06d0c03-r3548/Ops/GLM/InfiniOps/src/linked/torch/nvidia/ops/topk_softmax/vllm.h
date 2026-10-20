#ifndef INFINI_OPS_LINKED_TORCH_NVIDIA_OPS_TOPK_SOFTMAX_VLLM_H_
#define INFINI_OPS_LINKED_TORCH_NVIDIA_OPS_TOPK_SOFTMAX_VLLM_H_

#include <optional>

#include "linked/torch/ops/topk_softmax.h"
#include "torch/nvidia/c10.h"

namespace infini::ops::linked::torch::nvidia {

struct VllmTopkSoftmax : C10<Device::Type::kNvidia> {
  static void Call(at::Tensor topk_weights, at::Tensor topk_indices,
                   at::Tensor token_expert_indices, at::Tensor gating_output,
                   bool renormalize, std::optional<at::Tensor> bias);
};

}  // namespace infini::ops::linked::torch::nvidia

namespace infini::ops::linked::torch {

extern template class TorchTopkSoftmax<
    ::infini::ops::linked::torch::nvidia::VllmTopkSoftmax>;

}  // namespace infini::ops::linked::torch

namespace infini::ops {

template <>
class Operator<TopkSoftmax, Device::Type::kNvidia, 16>
    : public linked::torch::TorchTopkSoftmax<
          linked::torch::nvidia::VllmTopkSoftmax> {
 public:
  using linked::torch::TorchTopkSoftmax<
      linked::torch::nvidia::VllmTopkSoftmax>::TorchTopkSoftmax;

  using linked::torch::TorchTopkSoftmax<
      linked::torch::nvidia::VllmTopkSoftmax>::operator();
};

}  // namespace infini::ops

#endif  // INFINI_OPS_LINKED_TORCH_NVIDIA_OPS_TOPK_SOFTMAX_VLLM_H_
