#ifndef INFINI_OPS_NVIDIA_MOE_SUM_KERNEL_H_
#define INFINI_OPS_NVIDIA_MOE_SUM_KERNEL_H_

#include <optional>

#include "base/moe_sum.h"

namespace infini::ops {

template <>
class Operator<MoeSum, Device::Type::kNvidia, 0> : public MoeSum {
 public:
  using MoeSum::MoeSum;

  using MoeSum::operator();

  void operator()(const Tensor input, std::optional<Tensor> topk_ids,
                  std::optional<Tensor> expert_map,
                  Tensor output) const override;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_NVIDIA_MOE_SUM_KERNEL_H_
