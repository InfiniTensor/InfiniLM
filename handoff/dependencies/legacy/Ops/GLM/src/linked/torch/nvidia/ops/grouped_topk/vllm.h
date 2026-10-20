#ifndef INFINI_OPS_LINKED_TORCH_NVIDIA_OPS_GROUPED_TOPK_VLLM_H_
#define INFINI_OPS_LINKED_TORCH_NVIDIA_OPS_GROUPED_TOPK_VLLM_H_

#include <utility>

#include "linked/torch/ops/grouped_topk.h"
#include "torch/nvidia/c10.h"

namespace infini::ops::linked::torch::nvidia {

struct VllmGroupedTopk : C10<Device::Type::kNvidia> {
  static std::pair<at::Tensor, at::Tensor> Call(
      at::Tensor scores, at::Tensor bias, int64_t num_expert_group,
      int64_t topk_group, int64_t topk, bool renormalize,
      double routed_scaling_factor, int64_t scoring_func);
};

}  // namespace infini::ops::linked::torch::nvidia

namespace infini::ops::linked::torch {

extern template class TorchGroupedTopk<
    ::infini::ops::linked::torch::nvidia::VllmGroupedTopk>;

}  // namespace infini::ops::linked::torch

namespace infini::ops {

template <>
class Operator<GroupedTopk, Device::Type::kNvidia, 16>
    : public linked::torch::TorchGroupedTopk<
          linked::torch::nvidia::VllmGroupedTopk> {
 public:
  using linked::torch::TorchGroupedTopk<
      linked::torch::nvidia::VllmGroupedTopk>::TorchGroupedTopk;

  using linked::torch::TorchGroupedTopk<
      linked::torch::nvidia::VllmGroupedTopk>::operator();
};

}  // namespace infini::ops

#endif  // INFINI_OPS_LINKED_TORCH_NVIDIA_OPS_GROUPED_TOPK_VLLM_H_
