#ifndef INFINI_OPS_LINKED_TORCH_METAX_OPS_RMS_NORM_VLLM_H_
#define INFINI_OPS_LINKED_TORCH_METAX_OPS_RMS_NORM_VLLM_H_

#include "linked/torch/ops/rms_norm.h"
#include "torch/metax/c10.h"

namespace infini::ops::linked::torch::metax {

struct VllmRmsNorm : C10<Device::Type::kMetax> {
  static void Call(at::Tensor& out, at::Tensor& input, at::Tensor& weight,
                   double epsilon);
};

}  // namespace infini::ops::linked::torch::metax

namespace infini::ops::linked::torch {

extern template class TorchRmsNorm<
    ::infini::ops::linked::torch::metax::VllmRmsNorm>;

}  // namespace infini::ops::linked::torch

namespace infini::ops {

template <>
class Operator<RmsNorm, Device::Type::kMetax, 16>
    : public linked::torch::TorchRmsNorm<linked::torch::metax::VllmRmsNorm> {
 public:
  using linked::torch::TorchRmsNorm<
      linked::torch::metax::VllmRmsNorm>::TorchRmsNorm;
  using linked::torch::TorchRmsNorm<
      linked::torch::metax::VllmRmsNorm>::operator();
};

}  // namespace infini::ops

#endif  // INFINI_OPS_LINKED_TORCH_METAX_OPS_RMS_NORM_VLLM_H_
