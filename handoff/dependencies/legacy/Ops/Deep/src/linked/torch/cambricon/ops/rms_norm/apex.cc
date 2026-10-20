#include "linked/torch/cambricon/ops/rms_norm/apex.h"

// Cambricon Apex `fused_layer_norm_cuda` exports this global C++ symbol.
std::vector<at::Tensor> rms_norm_affine(const at::Tensor& input,
                                        const at::Tensor& bias,
                                        const at::Tensor& weight,
                                        double epsilon, long axis);

namespace infini::ops::linked::torch::cambricon {

void ApexRmsNorm::Call(at::Tensor& out, at::Tensor& input, at::Tensor& weight,
                       double epsilon) {
  const auto bias = at::empty({0}, at::TensorOptions().dtype(at::kFloat));
  const auto result =
      ::rms_norm_affine(input, bias, weight, epsilon, input.dim() - 1);
  out.copy_(result.at(0));
}

}  // namespace infini::ops::linked::torch::cambricon

namespace infini::ops::linked::torch {

template class TorchRmsNorm<
    ::infini::ops::linked::torch::cambricon::ApexRmsNorm>;

}  // namespace infini::ops::linked::torch
