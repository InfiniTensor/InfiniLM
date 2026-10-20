#ifndef INFINI_OPS_LINKED_TORCH_MOORE_OPS_SILU_AND_MUL_VLLM_H_
#define INFINI_OPS_LINKED_TORCH_MOORE_OPS_SILU_AND_MUL_VLLM_H_

#include "linked/torch/ops/silu_and_mul.h"
#include "torch/moore/c10.h"

namespace infini::ops::linked::torch::moore {

struct VllmSiluAndMul : C10<Device::Type::kMoore> {
  static void Call(at::Tensor& out, at::Tensor& input);
};

}  // namespace infini::ops::linked::torch::moore

namespace infini::ops::linked::torch {

extern template class TorchSiluAndMul<
    ::infini::ops::linked::torch::moore::VllmSiluAndMul>;

}  // namespace infini::ops::linked::torch

namespace infini::ops {

template <>
class Operator<SiluAndMul, Device::Type::kMoore, 16>
    : public linked::torch::TorchSiluAndMul<
          linked::torch::moore::VllmSiluAndMul> {
 public:
  using linked::torch::TorchSiluAndMul<
      linked::torch::moore::VllmSiluAndMul>::TorchSiluAndMul;
  using linked::torch::TorchSiluAndMul<
      linked::torch::moore::VllmSiluAndMul>::operator();
};

}  // namespace infini::ops

#endif  // INFINI_OPS_LINKED_TORCH_MOORE_OPS_SILU_AND_MUL_VLLM_H_
