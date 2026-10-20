#ifndef INFINI_OPS_LINKED_TORCH_NVIDIA_OPS_MOE_WNA16_MARLIN_GEMM_VLLM_H_
#define INFINI_OPS_LINKED_TORCH_NVIDIA_OPS_MOE_WNA16_MARLIN_GEMM_VLLM_H_

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/stack.h>
#include <c10/util/Exception.h>

#include <optional>
#include <utility>

#include "linked/torch/ops/moe_wna16_marlin_gemm.h"
#include "torch/nvidia/c10.h"

namespace infini::ops::linked::torch::nvidia {

struct VllmMoeWna16MarlinGemm : C10<Device::Type::kNvidia> {
  static void Validate(const DataType a_dtype, const int64_t b_type_id,
                       const bool has_a_scales, const bool has_global_scale) {
    constexpr int64_t kFloat4E2M1F = 562949953487106;
    const auto has_int8_activation = a_dtype == DataType::kInt8;
    TORCH_CHECK(
        a_dtype == DataType::kFloat16 || a_dtype == DataType::kBFloat16 ||
            has_int8_activation,
        "Linked `moe_wna16_marlin_gemm` supports float16, bfloat16, and "
        "int8 activations only; InfiniRT cannot represent vLLM's FP8 "
        "activation dtype.");
    TORCH_CHECK(has_a_scales == has_int8_activation,
                "Linked `moe_wna16_marlin_gemm` requires `a_scales` "
                "exactly when `a` is int8.");
    TORCH_CHECK(
        b_type_id != kFloat4E2M1F,
        "Linked `moe_wna16_marlin_gemm` does not support `float4_e2m1f` "
        "because InfiniRT cannot represent its float8 scales.");
    TORCH_CHECK(
        !has_global_scale,
        "Linked `moe_wna16_marlin_gemm` does not support `global_scale`.");
  }

  static void Call(at::Tensor a, at::Tensor out, at::Tensor b_q_weight,
                   std::optional<at::Tensor> b_bias_or_none,
                   at::Tensor b_scales, std::optional<at::Tensor> a_scales,
                   std::optional<at::Tensor> global_scale,
                   std::optional<at::Tensor> b_zeros_or_none,
                   std::optional<at::Tensor> g_idx_or_none,
                   std::optional<at::Tensor> perm_or_none, at::Tensor workspace,
                   at::Tensor sorted_token_ids, at::Tensor expert_ids,
                   at::Tensor num_tokens_past_padded, at::Tensor topk_weights,
                   int64_t moe_block_size, int64_t top_k, bool mul_topk_weights,
                   int64_t b_type_id, int64_t size_m, int64_t size_n,
                   int64_t size_k, bool is_full_k, bool use_atomic_add,
                   bool use_fp32_reduce, bool is_zp_float, int64_t thread_k,
                   int64_t thread_n, int64_t blocks_per_sm) {
    static const auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
        "_moe_C::moe_wna16_marlin_gemm", "");
    c10::Stack stack;
    stack.reserve(29);
    stack.emplace_back(std::move(a));
    stack.emplace_back(out);
    stack.emplace_back(std::move(b_q_weight));
    stack.emplace_back(b_bias_or_none ? c10::IValue(std::move(*b_bias_or_none))
                                      : c10::IValue());
    stack.emplace_back(std::move(b_scales));
    stack.emplace_back(a_scales ? c10::IValue(std::move(*a_scales))
                                : c10::IValue());
    stack.emplace_back(global_scale ? c10::IValue(std::move(*global_scale))
                                    : c10::IValue());
    stack.emplace_back(b_zeros_or_none
                           ? c10::IValue(std::move(*b_zeros_or_none))
                           : c10::IValue());
    stack.emplace_back(g_idx_or_none ? c10::IValue(std::move(*g_idx_or_none))
                                     : c10::IValue());
    stack.emplace_back(perm_or_none ? c10::IValue(std::move(*perm_or_none))
                                    : c10::IValue());
    stack.emplace_back(std::move(workspace));
    stack.emplace_back(std::move(sorted_token_ids));
    stack.emplace_back(std::move(expert_ids));
    stack.emplace_back(std::move(num_tokens_past_padded));
    stack.emplace_back(std::move(topk_weights));
    stack.emplace_back(moe_block_size);
    stack.emplace_back(top_k);
    stack.emplace_back(mul_topk_weights);
    stack.emplace_back(b_type_id);
    stack.emplace_back(size_m);
    stack.emplace_back(size_n);
    stack.emplace_back(size_k);
    stack.emplace_back(is_full_k);
    stack.emplace_back(use_atomic_add);
    stack.emplace_back(use_fp32_reduce);
    stack.emplace_back(is_zp_float);
    stack.emplace_back(thread_k);
    stack.emplace_back(thread_n);
    stack.emplace_back(blocks_per_sm);
    op.callBoxed(&stack);

    TORCH_CHECK(stack.size() == 1,
                "Linked `moe_wna16_marlin_gemm` returned an unexpected "
                "number of values.");
    auto result = std::move(stack.front()).toTensor();
    TORCH_CHECK(result.unsafeGetTensorImpl() == out.unsafeGetTensorImpl(),
                "Linked `moe_wna16_marlin_gemm` did not return the provided "
                "output tensor.");
  }
};

}  // namespace infini::ops::linked::torch::nvidia

namespace infini::ops::linked::torch {

extern template class TorchMoeWna16MarlinGemm<
    ::infini::ops::linked::torch::nvidia::VllmMoeWna16MarlinGemm>;

}  // namespace infini::ops::linked::torch

namespace infini::ops {

template <>
class Operator<MoeWna16MarlinGemm, Device::Type::kNvidia, 16>
    : public linked::torch::TorchMoeWna16MarlinGemm<
          linked::torch::nvidia::VllmMoeWna16MarlinGemm> {
 public:
  using linked::torch::TorchMoeWna16MarlinGemm<
      linked::torch::nvidia::VllmMoeWna16MarlinGemm>::TorchMoeWna16MarlinGemm;

  using linked::torch::TorchMoeWna16MarlinGemm<
      linked::torch::nvidia::VllmMoeWna16MarlinGemm>::operator();
};

}  // namespace infini::ops

#endif  // INFINI_OPS_LINKED_TORCH_NVIDIA_OPS_MOE_WNA16_MARLIN_GEMM_VLLM_H_
