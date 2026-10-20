#ifndef INFINI_OPS_LINKED_TORCH_THEAD_OPS_RMS_NORM_VLLM_H_
#define INFINI_OPS_LINKED_TORCH_THEAD_OPS_RMS_NORM_VLLM_H_

#include "linked/torch/ops/rms_norm.h"
#include "torch/thead/c10.h"

namespace infini::ops::linked::torch::thead {

struct VllmRmsNorm : C10<Device::Type::kThead> {
  static void Call(at::Tensor& out, at::Tensor& input, at::Tensor& weight,
                   double epsilon);
};

}  // namespace infini::ops::linked::torch::thead

namespace infini::ops::linked::torch {

extern template class TorchRmsNorm<
    ::infini::ops::linked::torch::thead::VllmRmsNorm>;

}  // namespace infini::ops::linked::torch

namespace infini::ops {

template <>
class Operator<RmsNorm, Device::Type::kThead, 16>
    : public linked::torch::TorchRmsNorm<linked::torch::thead::VllmRmsNorm> {
 public:
  using linked::torch::TorchRmsNorm<
      linked::torch::thead::VllmRmsNorm>::TorchRmsNorm;
  using linked::torch::TorchRmsNorm<
      linked::torch::thead::VllmRmsNorm>::operator();
};

}  // namespace infini::ops

#endif  // INFINI_OPS_LINKED_TORCH_THEAD_OPS_RMS_NORM_VLLM_H_
