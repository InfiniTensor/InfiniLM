#include "linked/torch/nvidia/ops/gptq_marlin_repack/vllm.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/stack.h>
#include <c10/core/SymInt.h>

#include <cassert>
#include <utility>

namespace infini::ops::linked::torch::nvidia {

at::Tensor VllmGptqMarlinRepack::Call(at::Tensor b_q_weight, at::Tensor perm,
                                      int64_t size_k, int64_t size_n,
                                      int64_t num_bits, bool is_a_8bit) {
  static const auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
      "_C::gptq_marlin_repack", "");
  c10::Stack stack;
  stack.emplace_back(std::move(b_q_weight));
  stack.emplace_back(std::move(perm));
  stack.emplace_back(c10::SymInt{size_k});
  stack.emplace_back(c10::SymInt{size_n});
  stack.emplace_back(num_bits);
  stack.emplace_back(is_a_8bit);
  op.callBoxed(&stack);

  assert(stack.size() == 1 &&
         "`gptq_marlin_repack` returned an unexpected number of values");
  return std::move(stack.front()).toTensor();
}

}  // namespace infini::ops::linked::torch::nvidia

namespace infini::ops::linked::torch {

template class TorchGptqMarlinRepack<
    ::infini::ops::linked::torch::nvidia::VllmGptqMarlinRepack>;

}  // namespace infini::ops::linked::torch
