#ifndef INFINI_OPS_LINKED_TORCH_OPS_FLASH_ATTN_VARLEN_FUNC_H_
#define INFINI_OPS_LINKED_TORCH_OPS_FLASH_ATTN_VARLEN_FUNC_H_

#include <ATen/core/Generator.h>

#include <cassert>
#include <cmath>
#include <optional>
#include <vector>

#include "base/flash_attn_varlen_func.h"
#include "torch/tensor_.h"

namespace infini::ops::linked::torch {

template <typename Backend>
class TorchFlashAttnVarlenFunc : public ::infini::ops::FlashAttnVarlenFunc {
 public:
  using ::infini::ops::FlashAttnVarlenFunc::FlashAttnVarlenFunc;

  using ::infini::ops::FlashAttnVarlenFunc::operator();

  void operator()(const Tensor q, const Tensor k, const Tensor v,
                  const Tensor cu_seqlens_q, const Tensor cu_seqlens_k,
                  const std::optional<Tensor> alibi_slopes,
                  const std::optional<Tensor> block_table,
                  const int64_t max_seqlen_q, const int64_t max_seqlen_k,
                  const double dropout_p,
                  const std::optional<double> softmax_scale, const bool causal,
                  const std::vector<int64_t> window_size, const double softcap,
                  const bool deterministic, const bool return_attn_probs,
                  Tensor out, std::optional<Tensor> softmax_lse,
                  std::optional<Tensor> s_dmask) const override {
    const typename Backend::StreamGuard stream_guard{
        Backend::GetStreamFromExternal(stream_, device_index_)};

    auto at_q = ToAtenTensor<Backend::kDeviceType>(const_cast<void*>(q.data()),
                                                   q_shape_, q_strides_,
                                                   q_dtype_, device_index_);
    auto at_k = ToAtenTensor<Backend::kDeviceType>(const_cast<void*>(k.data()),
                                                   k_shape_, k_strides_,
                                                   k_dtype_, device_index_);
    auto at_v = ToAtenTensor<Backend::kDeviceType>(const_cast<void*>(v.data()),
                                                   v_shape_, v_strides_,
                                                   v_dtype_, device_index_);
    auto at_cu_seqlens_q = ToAtenTensor<Backend::kDeviceType>(
        const_cast<void*>(cu_seqlens_q.data()), cu_seqlens_q_shape_,
        cu_seqlens_q_strides_, cu_seqlens_q_dtype_, device_index_);
    auto at_cu_seqlens_k = ToAtenTensor<Backend::kDeviceType>(
        const_cast<void*>(cu_seqlens_k.data()), cu_seqlens_k_shape_,
        cu_seqlens_k_strides_, cu_seqlens_k_dtype_, device_index_);
    auto at_out = ToAtenTensor<Backend::kDeviceType>(
        out.data(), out_shape_, out_strides_, out_dtype_, device_index_);

    std::optional<at::Tensor> at_alibi_slopes;
    std::optional<at::Tensor> at_block_table;
    std::optional<at::Tensor> at_softmax_lse;
    std::optional<at::Tensor> at_s_dmask;

    if (alibi_slopes.has_value()) {
      at_alibi_slopes.emplace(ToAtenTensor<Backend::kDeviceType>(
          const_cast<void*>(alibi_slopes->data()), alibi_slopes_shape_,
          alibi_slopes_strides_, alibi_slopes_dtype_, device_index_));
    }
    if (block_table.has_value()) {
      at_block_table.emplace(ToAtenTensor<Backend::kDeviceType>(
          const_cast<void*>(block_table->data()), block_table_shape_,
          block_table_strides_, block_table_dtype_, device_index_));
    }
    if (softmax_lse.has_value()) {
      at_softmax_lse.emplace(ToAtenTensor<Backend::kDeviceType>(
          softmax_lse->data(), softmax_lse_shape_, softmax_lse_strides_,
          softmax_lse_dtype_, device_index_));
      at_s_dmask.emplace(ToAtenTensor<Backend::kDeviceType>(
          s_dmask->data(), s_dmask_shape_, s_dmask_strides_, s_dmask_dtype_,
          device_index_));
    }

    std::optional<at::Tensor> at_out_optional;
    std::optional<at::Tensor> at_seqused_k;
    std::optional<const at::Tensor> at_leftpad_k;
    std::optional<at::Generator> generator;
    auto result = Backend::Call(
        at_q, at_k, at_v, at_out_optional, at_cu_seqlens_q, at_cu_seqlens_k,
        at_seqused_k, at_leftpad_k, at_block_table, at_alibi_slopes,
        static_cast<int>(max_seqlen_q), static_cast<int>(max_seqlen_k),
        static_cast<float>(dropout_p),
        static_cast<float>(softmax_scale.value_or(
            1.0 / std::sqrt(static_cast<double>(q_shape_[2])))),
        false, causal, static_cast<int>(window_size[0]),
        static_cast<int>(window_size[1]), static_cast<float>(softcap),
        return_attn_probs && dropout_p > 0.0, generator);
    assert(!result.empty() &&
           "Linked `flash_attn_varlen_func` provider returned no output.");
    at_out.copy_(result[0]);
    if (at_softmax_lse.has_value()) {
      assert(result.size() >= 3 &&
             "Linked `flash_attn_varlen_func` provider did not return "
             "auxiliary outputs.");
      at_softmax_lse->copy_(result[1]);
      at_s_dmask->copy_(result[2]);
    }

    (void)deterministic;
  }
};

}  // namespace infini::ops::linked::torch

#endif  // INFINI_OPS_LINKED_TORCH_OPS_FLASH_ATTN_VARLEN_FUNC_H_
