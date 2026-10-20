#include "linked/torch/moore/ops/silu_and_mul/vllm.h"

// Moore vLLM `_C` exports this global C++ symbol.
void silu_and_mul(at::Tensor& out, at::Tensor& input);

namespace infini::ops::linked::torch::moore {

void VllmSiluAndMul::Call(at::Tensor& out, at::Tensor& input) {
  ::silu_and_mul(out, input);
}

}  // namespace infini::ops::linked::torch::moore

namespace infini::ops::linked::torch {

template class TorchSiluAndMul<
    ::infini::ops::linked::torch::moore::VllmSiluAndMul>;

}  // namespace infini::ops::linked::torch
