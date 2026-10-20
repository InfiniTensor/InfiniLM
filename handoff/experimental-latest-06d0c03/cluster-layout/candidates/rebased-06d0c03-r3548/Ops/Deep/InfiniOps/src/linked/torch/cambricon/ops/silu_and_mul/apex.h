#ifndef INFINI_OPS_LINKED_TORCH_CAMBRICON_OPS_SILU_AND_MUL_APEX_H_
#define INFINI_OPS_LINKED_TORCH_CAMBRICON_OPS_SILU_AND_MUL_APEX_H_

#include "linked/torch/ops/silu_and_mul.h"
#include "torch/cambricon/c10.h"

namespace infini::ops::linked::torch::cambricon {

struct ApexSiluAndMul : C10<Device::Type::kCambricon> {
  static void Call(at::Tensor& out, at::Tensor& input);
};

}  // namespace infini::ops::linked::torch::cambricon

namespace infini::ops::linked::torch {

extern template class TorchSiluAndMul<
    ::infini::ops::linked::torch::cambricon::ApexSiluAndMul>;

}  // namespace infini::ops::linked::torch

namespace infini::ops {

template <>
class Operator<SiluAndMul, Device::Type::kCambricon, 16>
    : public linked::torch::TorchSiluAndMul<
          linked::torch::cambricon::ApexSiluAndMul> {
 public:
  using linked::torch::TorchSiluAndMul<
      linked::torch::cambricon::ApexSiluAndMul>::TorchSiluAndMul;
  using linked::torch::TorchSiluAndMul<
      linked::torch::cambricon::ApexSiluAndMul>::operator();
};

}  // namespace infini::ops

#endif  // INFINI_OPS_LINKED_TORCH_CAMBRICON_OPS_SILU_AND_MUL_APEX_H_
