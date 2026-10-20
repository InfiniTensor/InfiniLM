#include "linked/torch/nvidia/ops/grouped_topk/vllm.h"

#include <ATen/Functions.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/stack.h>
#include <c10/util/Exception.h>

#include <utility>

namespace infini::ops::linked::torch::nvidia {

std::pair<at::Tensor, at::Tensor> VllmGroupedTopk::Call(
    at::Tensor scores, at::Tensor bias, int64_t num_expert_group,
    int64_t topk_group, int64_t topk, bool renormalize,
    double routed_scaling_factor, int64_t scoring_func) {
  TORCH_CHECK(scoring_func == 0 || scoring_func == 1,
              "Linked vLLM `grouped_topk` requires `scoring_func` 0 (none) "
              "or 1 (sigmoid).");

  auto routed_scores = scoring_func == 0 ? scores : at::sigmoid(scores);
  auto scores_with_bias =
      (routed_scores + bias.unsqueeze(0)).to(routed_scores.scalar_type());

  static const auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
      "_moe_C::grouped_topk", "");
  c10::Stack stack;
  stack.emplace_back(std::move(routed_scores));
  stack.emplace_back(std::move(scores_with_bias));
  stack.emplace_back(num_expert_group);
  stack.emplace_back(topk_group);
  stack.emplace_back(topk);
  stack.emplace_back(renormalize);
  stack.emplace_back(routed_scaling_factor);
  op.callBoxed(&stack);

  TORCH_CHECK(stack.size() == 2,
              "Linked vLLM `grouped_topk` returned an unexpected number of "
              "values.");
  return {std::move(stack[0]).toTensor(), std::move(stack[1]).toTensor()};
}

}  // namespace infini::ops::linked::torch::nvidia

namespace infini::ops::linked::torch {

template class TorchGroupedTopk<
    ::infini::ops::linked::torch::nvidia::VllmGroupedTopk>;

}  // namespace infini::ops::linked::torch
