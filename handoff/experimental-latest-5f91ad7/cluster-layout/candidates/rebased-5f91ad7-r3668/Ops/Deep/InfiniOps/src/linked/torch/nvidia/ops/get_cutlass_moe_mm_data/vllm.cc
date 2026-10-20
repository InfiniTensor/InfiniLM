#include "linked/torch/nvidia/ops/get_cutlass_moe_mm_data/vllm.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/stack.h>
#include <cuda_runtime_api.h>

#include <cassert>
#include <utility>

namespace infini::ops::linked::torch::nvidia {

void VllmGetCutlassMoeMmData::ValidateDevice(int device_index) {
  cudaDeviceProp properties{};
  const auto status = cudaGetDeviceProperties(&properties, device_index);
  TORCH_CHECK(status == cudaSuccess,
              "Failed to query the CUDA device for linked "
              "`get_cutlass_moe_mm_data`.");
  TORCH_CHECK(
      properties.major >= 9,
      "The linked vLLM `get_cutlass_moe_mm_data` implementation requires "
      "compute capability 9.0 or newer.");
}

void VllmGetCutlassMoeMmData::Call(
    at::Tensor topk_ids, at::Tensor expert_offsets, at::Tensor problem_sizes1,
    at::Tensor problem_sizes2, at::Tensor input_permutation,
    at::Tensor output_permutation, int64_t num_experts, int64_t n, int64_t k,
    std::optional<at::Tensor> blockscale_offsets) {
  static const auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
      "_C::get_cutlass_moe_mm_data", "");
  c10::Stack stack;
  stack.reserve(10);
  stack.emplace_back(std::move(topk_ids));
  stack.emplace_back(std::move(expert_offsets));
  stack.emplace_back(std::move(problem_sizes1));
  stack.emplace_back(std::move(problem_sizes2));
  stack.emplace_back(std::move(input_permutation));
  stack.emplace_back(std::move(output_permutation));
  stack.emplace_back(num_experts);
  stack.emplace_back(n);
  stack.emplace_back(k);
  if (blockscale_offsets.has_value()) {
    stack.emplace_back(std::move(*blockscale_offsets));
  } else {
    stack.emplace_back();
  }
  op.callBoxed(&stack);

  assert(stack.empty() &&
         "`get_cutlass_moe_mm_data` returned an unexpected value.");
}

}  // namespace infini::ops::linked::torch::nvidia

namespace infini::ops::linked::torch {

template class TorchGetCutlassMoeMmData<
    ::infini::ops::linked::torch::nvidia::VllmGetCutlassMoeMmData>;

}  // namespace infini::ops::linked::torch
