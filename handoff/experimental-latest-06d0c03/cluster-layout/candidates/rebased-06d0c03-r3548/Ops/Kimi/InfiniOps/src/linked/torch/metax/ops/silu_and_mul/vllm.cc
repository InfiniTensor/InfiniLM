#include "linked/torch/metax/ops/silu_and_mul/vllm.h"

// MetaX vLLM `_C` exports this global C++ symbol.
void silu_and_mul(at::Tensor& out, at::Tensor& input);

namespace infini::ops::linked::torch::metax {

void VllmSiluAndMul::Call(at::Tensor& out, at::Tensor& input) {
  ::silu_and_mul(out, input);
}

}  // namespace infini::ops::linked::torch::metax

namespace infini::ops::linked::torch {

template class TorchSiluAndMul<
    ::infini::ops::linked::torch::metax::VllmSiluAndMul>;

}  // namespace infini::ops::linked::torch
