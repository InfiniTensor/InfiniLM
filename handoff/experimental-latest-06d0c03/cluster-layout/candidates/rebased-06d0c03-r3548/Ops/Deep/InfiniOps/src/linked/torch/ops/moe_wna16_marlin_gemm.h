#ifndef INFINI_OPS_LINKED_TORCH_OPS_MOE_WNA16_MARLIN_GEMM_H_
#define INFINI_OPS_LINKED_TORCH_OPS_MOE_WNA16_MARLIN_GEMM_H_

#include <optional>
#include <utility>

#include "base/moe_wna16_marlin_gemm.h"
#include "torch/tensor_.h"

namespace infini::ops::linked::torch {

template <typename Backend>
class TorchMoeWna16MarlinGemm : public ::infini::ops::MoeWna16MarlinGemm {
 public:
  using ::infini::ops::MoeWna16MarlinGemm::MoeWna16MarlinGemm;

  using ::infini::ops::MoeWna16MarlinGemm::operator();

  void operator()(const Tensor a, const Tensor b_q_weight,
                  std::optional<Tensor> b_bias_or_none, const Tensor b_scales,
                  std::optional<Tensor> a_scales,
                  std::optional<Tensor> global_scale,
                  std::optional<Tensor> b_zeros_or_none,
                  std::optional<Tensor> g_idx_or_none,
                  std::optional<Tensor> perm_or_none, const Tensor workspace,
                  const Tensor sorted_token_ids, const Tensor expert_ids,
                  const Tensor num_tokens_past_padded,
                  const Tensor topk_weights, const int64_t moe_block_size,
                  const int64_t top_k, const bool mul_topk_weights,
                  const int64_t b_type_id, const int64_t size_m,
                  const int64_t size_n, const int64_t size_k,
                  const bool is_full_k, const bool use_atomic_add,
                  const bool use_fp32_reduce, const bool is_zp_float,
                  const int64_t thread_k, const int64_t thread_n,
                  const int64_t blocks_per_sm, Tensor out) const override {
    ValidateCallMetadata(a, b_q_weight, b_bias_or_none, b_scales, a_scales,
                         global_scale, b_zeros_or_none, g_idx_or_none,
                         perm_or_none, workspace, sorted_token_ids, expert_ids,
                         num_tokens_past_padded, topk_weights, moe_block_size,
                         top_k, mul_topk_weights, b_type_id, size_m, size_n,
                         size_k, is_full_k, use_atomic_add, use_fp32_reduce,
                         is_zp_float, thread_k, thread_n, blocks_per_sm, out);

    Backend::Validate(a.dtype(), b_type_id, a_scales.has_value(),
                      global_scale.has_value());

    const typename Backend::StreamGuard stream_guard{
        Backend::GetStreamFromExternal(stream_, device_index_)};
    Backend::Call(ToAten(a), ToAten(out), ToAten(b_q_weight),
                  ToOptionalAten(b_bias_or_none), ToAten(b_scales),
                  ToOptionalAten(a_scales), ToOptionalAten(global_scale),
                  ToOptionalAten(b_zeros_or_none),
                  ToOptionalAten(g_idx_or_none), ToOptionalAten(perm_or_none),
                  ToAten(workspace), ToAten(sorted_token_ids),
                  ToAten(expert_ids), ToAten(num_tokens_past_padded),
                  ToAten(topk_weights), moe_block_size, top_k, mul_topk_weights,
                  b_type_id, size_m, size_n, size_k, is_full_k, use_atomic_add,
                  use_fp32_reduce, is_zp_float, thread_k, thread_n,
                  blocks_per_sm);
  }

 private:
  at::Tensor ToAten(const Tensor tensor) const {
    return ToAtenTensor<Backend::kDeviceType>(const_cast<void*>(tensor.data()),
                                              tensor.shape(), tensor.strides(),
                                              tensor.dtype(), device_index_);
  }

  std::optional<at::Tensor> ToOptionalAten(
      const std::optional<Tensor>& tensor) const {
    if (!tensor) {
      return std::nullopt;
    }

    return ToAten(*tensor);
  }
};

}  // namespace infini::ops::linked::torch

#endif  // INFINI_OPS_LINKED_TORCH_OPS_MOE_WNA16_MARLIN_GEMM_H_
