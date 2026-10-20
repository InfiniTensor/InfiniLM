#ifndef INFINI_OPS_LINKED_TORCH_METAX_OPS_SILU_AND_MUL_VLLM_H_
#define INFINI_OPS_LINKED_TORCH_METAX_OPS_SILU_AND_MUL_VLLM_H_

#include "linked/torch/ops/silu_and_mul.h"
#include "torch/metax/c10.h"

namespace infini::ops::linked::torch::metax {

struct VllmSiluAndMul : C10<Device::Type::kMetax> {
  static void Call(at::Tensor& out, at::Tensor& input);
};

}  // namespace infini::ops::linked::torch::metax

namespace infini::ops::linked::torch {

extern template class TorchSiluAndMul<
    ::infini::ops::linked::torch::metax::VllmSiluAndMul>;

}  // namespace infini::ops::linked::torch

namespace infini::ops {

template <>
class Operator<SiluAndMul, Device::Type::kMetax, 16>
    : public linked::torch::TorchSiluAndMul<
          linked::torch::metax::VllmSiluAndMul> {
 public:
  using linked::torch::TorchSiluAndMul<
      linked::torch::metax::VllmSiluAndMul>::TorchSiluAndMul;
  using linked::torch::TorchSiluAndMul<
      linked::torch::metax::VllmSiluAndMul>::operator();
};

}  // namespace infini::ops

#endif  // INFINI_OPS_LINKED_TORCH_METAX_OPS_SILU_AND_MUL_VLLM_H_
