#include "linked/torch/thead/ops/rms_norm/vllm.h"

// T-Head PPU vLLM `_C` exports this global C++ symbol.
void rms_norm(at::Tensor& out, at::Tensor& input, at::Tensor& weight,
              double epsilon);

namespace infini::ops::linked::torch::thead {

void VllmRmsNorm::Call(at::Tensor& out, at::Tensor& input, at::Tensor& weight,
                       double epsilon) {
  ::rms_norm(out, input, weight, epsilon);
}

}  // namespace infini::ops::linked::torch::thead

namespace infini::ops::linked::torch {

template class TorchRmsNorm< ::infini::ops::linked::torch::thead::VllmRmsNorm>;

}  // namespace infini::ops::linked::torch
