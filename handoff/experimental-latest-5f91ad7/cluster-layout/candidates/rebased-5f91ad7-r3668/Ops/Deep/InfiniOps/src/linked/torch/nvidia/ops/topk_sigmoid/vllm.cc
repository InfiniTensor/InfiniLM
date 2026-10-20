#include "linked/torch/nvidia/ops/topk_sigmoid/vllm.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/stack.h>
#include <c10/util/Exception.h>

#include <utility>

namespace infini::ops::linked::torch::nvidia {

void VllmTopkSigmoid::Validate(bool has_is_padding,
                               double routed_scaling_factor) {
  TORCH_CHECK(!has_is_padding,
              "Linked `topk_sigmoid` does not support `is_padding`.");
  TORCH_CHECK(routed_scaling_factor == 1.0,
              "Linked `topk_sigmoid` requires "
              "`routed_scaling_factor == 1.0`.");
}

void VllmTopkSigmoid::Call(at::Tensor topk_weights, at::Tensor topk_indices,
                           at::Tensor token_expert_indices,
                           at::Tensor gating_output, bool renormalize,
                           std::optional<at::Tensor> bias) {
  static const auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
      "_moe_C::topk_sigmoid", "");
  c10::Stack stack;
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

  TORCH_CHECK(stack.empty(),
              "Linked `topk_sigmoid` returned unexpected values.");
}

}  // namespace infini::ops::linked::torch::nvidia

namespace infini::ops::linked::torch {

template class TorchTopkSigmoid<
    ::infini::ops::linked::torch::nvidia::VllmTopkSigmoid>;

}  // namespace infini::ops::linked::torch
