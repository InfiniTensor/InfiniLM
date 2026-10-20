#include "linked/torch/nvidia/ops/topk_softmax/vllm.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/stack.h>

#include <cassert>
#include <utility>

namespace infini::ops::linked::torch::nvidia {

void VllmTopkSoftmax::Call(at::Tensor topk_weights, at::Tensor topk_indices,
                           at::Tensor token_expert_indices,
                           at::Tensor gating_output, bool renormalize,
                           std::optional<at::Tensor> bias) {
  static const auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
      "_moe_C::topk_softmax", "");
  c10::Stack stack;
  stack.reserve(6);
  stack.emplace_back(std::move(topk_weights));
  stack.emplace_back(std::move(topk_indices));
  stack.emplace_back(std::move(token_expert_indices));
  stack.emplace_back(std::move(gating_output));
  stack.emplace_back(renormalize);
  if (bias.has_value()) {
    stack.emplace_back(std::move(*bias));
  } else {
    stack.emplace_back();
  }
  op.callBoxed(&stack);

  assert(stack.empty() &&
         "`topk_softmax` returned an unexpected number of values");
}

}  // namespace infini::ops::linked::torch::nvidia

namespace infini::ops::linked::torch {

template class TorchTopkSoftmax<
    ::infini::ops::linked::torch::nvidia::VllmTopkSoftmax>;

}  // namespace infini::ops::linked::torch
