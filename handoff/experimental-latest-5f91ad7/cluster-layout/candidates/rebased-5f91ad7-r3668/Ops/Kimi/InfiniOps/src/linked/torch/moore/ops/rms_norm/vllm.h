#ifndef INFINI_OPS_LINKED_TORCH_MOORE_OPS_RMS_NORM_VLLM_H_
#define INFINI_OPS_LINKED_TORCH_MOORE_OPS_RMS_NORM_VLLM_H_

#include "linked/torch/ops/rms_norm.h"
#include "torch/moore/c10.h"

namespace infini::ops::linked::torch::moore {

struct VllmRmsNorm : C10<Device::Type::kMoore> {
  static void Call(at::Tensor& out, at::Tensor& input, at::Tensor& weight,
                   double epsilon);
};

}  // namespace infini::ops::linked::torch::moore

namespace infini::ops::linked::torch {

extern template class TorchRmsNorm<
    ::infini::ops::linked::torch::moore::VllmRmsNorm>;

}  // namespace infini::ops::linked::torch

namespace infini::ops {

template <>
class Operator<RmsNorm, Device::Type::kMoore, 16>
    : public linked::torch::TorchRmsNorm<linked::torch::moore::VllmRmsNorm> {
 public:
  using linked::torch::TorchRmsNorm<
      linked::torch::moore::VllmRmsNorm>::TorchRmsNorm;
  using linked::torch::TorchRmsNorm<
      linked::torch::moore::VllmRmsNorm>::operator();
};

}  // namespace infini::ops

#endif  // INFINI_OPS_LINKED_TORCH_MOORE_OPS_RMS_NORM_VLLM_H_
