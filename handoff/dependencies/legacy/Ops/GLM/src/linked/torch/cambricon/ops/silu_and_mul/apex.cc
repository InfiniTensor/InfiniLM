#include "linked/torch/cambricon/ops/silu_and_mul/apex.h"

// Cambricon Apex glu_activation exports this global C++ symbol.
at::Tensor bias_swiglu_fwd(const at::Tensor& input, const at::Tensor& bias);

namespace infini::ops::linked::torch::cambricon {

void ApexSiluAndMul::Call(at::Tensor& out, at::Tensor& input) {
  const auto bias = at::empty({0}, at::TensorOptions().dtype(at::kInt));
  out.copy_(::bias_swiglu_fwd(input, bias));
}

}  // namespace infini::ops::linked::torch::cambricon

namespace infini::ops::linked::torch {

template class TorchSiluAndMul<
    ::infini::ops::linked::torch::cambricon::ApexSiluAndMul>;

}  // namespace infini::ops::linked::torch
