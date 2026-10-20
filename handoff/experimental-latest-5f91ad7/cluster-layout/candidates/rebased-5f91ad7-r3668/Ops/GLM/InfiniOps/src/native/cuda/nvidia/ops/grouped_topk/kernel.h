#ifndef INFINI_OPS_NVIDIA_GROUPED_TOPK_KERNEL_H_
#define INFINI_OPS_NVIDIA_GROUPED_TOPK_KERNEL_H_

#include "base/grouped_topk.h"

namespace infini::ops {

template <>
class Operator<GroupedTopk, Device::Type::kNvidia, 0> : public GroupedTopk {
 public:
  using GroupedTopk::GroupedTopk;

  using GroupedTopk::operator();

  void operator()(const Tensor scores, const Tensor bias,
                  const int64_t num_expert_group, const int64_t topk_group,
                  const int64_t topk, const bool renormalize,
                  const double routed_scaling_factor,
                  const int64_t scoring_func, Tensor topk_values,
                  Tensor topk_indices) const override;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_NVIDIA_GROUPED_TOPK_KERNEL_H_
