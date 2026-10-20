#ifndef INFINI_OPS_LINKED_TORCH_CAMBRICON_OPS_RMS_NORM_APEX_H_
#define INFINI_OPS_LINKED_TORCH_CAMBRICON_OPS_RMS_NORM_APEX_H_

#include "linked/torch/ops/rms_norm.h"
#include "torch/cambricon/c10.h"

namespace infini::ops::linked::torch::cambricon {

struct ApexRmsNorm : C10<Device::Type::kCambricon> {
  static void Call(at::Tensor& out, at::Tensor& input, at::Tensor& weight,
                   double epsilon);
};

}  // namespace infini::ops::linked::torch::cambricon

namespace infini::ops::linked::torch {

extern template class TorchRmsNorm<
    ::infini::ops::linked::torch::cambricon::ApexRmsNorm>;

}  // namespace infini::ops::linked::torch

namespace infini::ops {

template <>
class Operator<RmsNorm, Device::Type::kCambricon, 16>
    : public linked::torch::TorchRmsNorm<
          linked::torch::cambricon::ApexRmsNorm> {
 public:
  using linked::torch::TorchRmsNorm<
      linked::torch::cambricon::ApexRmsNorm>::TorchRmsNorm;
  using linked::torch::TorchRmsNorm<
      linked::torch::cambricon::ApexRmsNorm>::operator();
};

}  // namespace infini::ops

#endif  // INFINI_OPS_LINKED_TORCH_CAMBRICON_OPS_RMS_NORM_APEX_H_
