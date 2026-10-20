#include "linked/torch/nvidia/ops/moe_wna16_gemm/vllm.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/stack.h>
#include <c10/util/Exception.h>

#include <utility>

namespace infini::ops::linked::torch::nvidia {

void VllmMoeWna16Gemm::Call(at::Tensor input, at::Tensor output,
                            at::Tensor b_qweight, at::Tensor b_scales,
                            std::optional<at::Tensor> b_qzeros,
                            std::optional<at::Tensor> topk_weights,
                            at::Tensor sorted_token_ids, at::Tensor expert_ids,
                            at::Tensor num_tokens_post_pad, int64_t top_k,
                            int64_t block_size_m, int64_t block_size_n,
                            int64_t block_size_k, int64_t bit) {
  static const auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
      "_moe_C::moe_wna16_gemm", "");
  c10::Stack stack;
  stack.reserve(14);
  stack.emplace_back(std::move(input));
  stack.emplace_back(output);
  stack.emplace_back(std::move(b_qweight));
  stack.emplace_back(std::move(b_scales));
  stack.emplace_back(b_qzeros ? c10::IValue(std::move(*b_qzeros))
                              : c10::IValue());
  stack.emplace_back(topk_weights ? c10::IValue(std::move(*topk_weights))
                                  : c10::IValue());
  stack.emplace_back(std::move(sorted_token_ids));
  stack.emplace_back(std::move(expert_ids));
  stack.emplace_back(std::move(num_tokens_post_pad));
  stack.emplace_back(top_k);
  stack.emplace_back(block_size_m);
  stack.emplace_back(block_size_n);
  stack.emplace_back(block_size_k);
  stack.emplace_back(bit);
  op.callBoxed(&stack);

  TORCH_CHECK(stack.size() == 1,
              "`moe_wna16_gemm` returned an unexpected number of values");
  (void)std::move(stack.front()).toTensor();
}

}  // namespace infini::ops::linked::torch::nvidia

namespace infini::ops::linked::torch {

template class TorchMoeWna16Gemm<
    ::infini::ops::linked::torch::nvidia::VllmMoeWna16Gemm>;

}  // namespace infini::ops::linked::torch
