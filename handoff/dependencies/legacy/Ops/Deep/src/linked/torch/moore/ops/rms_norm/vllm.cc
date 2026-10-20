#include "linked/torch/moore/ops/rms_norm/vllm.h"

// Moore vLLM `_C` exports this global C++ symbol.
void rms_norm(at::Tensor& out, at::Tensor& input, at::Tensor& weight,
              double epsilon);

namespace infini::ops::linked::torch::moore {

void VllmRmsNorm::Call(at::Tensor& out, at::Tensor& input, at::Tensor& weight,
                       double epsilon) {
  ::rms_norm(out, input, weight, epsilon);
}

}  // namespace infini::ops::linked::torch::moore

namespace infini::ops::linked::torch {

template class TorchRmsNorm< ::infini::ops::linked::torch::moore::VllmRmsNorm>;

}  // namespace infini::ops::linked::torch
