#include "linked/torch/thead/ops/silu_and_mul/vllm.h"

// T-Head PPU vLLM `_C` exports this global C++ symbol.
void silu_and_mul(at::Tensor& out, at::Tensor& input);

namespace infini::ops::linked::torch::thead {

void VllmSiluAndMul::Call(at::Tensor& out, at::Tensor& input) {
  ::silu_and_mul(out, input);
}

}  // namespace infini::ops::linked::torch::thead

namespace infini::ops::linked::torch {

template class TorchSiluAndMul<
    ::infini::ops::linked::torch::thead::VllmSiluAndMul>;

}  // namespace infini::ops::linked::torch
