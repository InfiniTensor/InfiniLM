#ifndef INFINI_OPS_TORCH_FLASH_ATTN_VARLEN_FUNC_ATEN_IMPL_H_
#define INFINI_OPS_TORCH_FLASH_ATTN_VARLEN_FUNC_ATEN_IMPL_H_

#include <ATen/ops/_flash_attention_forward.h>
#include <ATen/ops/arange.h>

#include <cassert>
#include <tuple>

#include "torch/c10.h"
#include "torch/ops/flash_attn_varlen_func/flash_attn_varlen_func.h"
#include "torch/tensor_.h"

namespace infini::ops {

namespace detail {

struct AtenFlashAttnVarlenOptions {
  std::optional<int64_t> window_size_left;
  std::optional<int64_t> window_size_right;
};

template <Device::Type kDev>
struct AtenFlashAttnVarlenPolicy;

}  // namespace detail

template <Device::Type kDev>
void AtenFlashAttnVarlenFunc<kDev>::operator()(
    const Tensor q, const Tensor k, const Tensor v, const Tensor cu_seqlens_q,
    const Tensor cu_seqlens_k, const std::optional<Tensor> alibi_slopes,
    const std::optional<Tensor> block_table, const int64_t max_seqlen_q,
    const int64_t max_seqlen_k, const double dropout_p,
    const std::optional<double> softmax_scale, const bool causal,
    const std::vector<int64_t> window_size, const double softcap,
    const bool deterministic, const bool return_attn_probs, Tensor out,
    std::optional<Tensor> softmax_lse, std::optional<Tensor> s_dmask) const {
  assert(!alibi_slopes.has_value() &&
         "The PyTorch `FlashAttnVarlenFunc` provider does not support "
         "`alibi_slopes`.");
  assert(!block_table.has_value() &&
         "The PyTorch `FlashAttnVarlenFunc` provider does not support "
         "`block_table`.");

  using Policy = detail::AtenFlashAttnVarlenPolicy<kDev>;
  Policy::ValidateWindow(window_size);

  (void)softcap;
  (void)deterministic;
  (void)return_attn_probs;

  const auto run = [&]() {
    auto at_q = ToAtenTensor<kDev>(const_cast<void*>(q.data()), q_shape_,
                                   q_strides_, q_dtype_, device_index_);
    auto at_k = ToAtenTensor<kDev>(const_cast<void*>(k.data()), k_shape_,
                                   k_strides_, k_dtype_, device_index_);
    auto at_v = ToAtenTensor<kDev>(const_cast<void*>(v.data()), v_shape_,
                                   v_strides_, v_dtype_, device_index_);
    auto at_cu_seqlens_q = ToAtenTensor<kDev>(
        const_cast<void*>(cu_seqlens_q.data()), cu_seqlens_q_shape_,
        cu_seqlens_q_strides_, cu_seqlens_q_dtype_, device_index_);
    auto at_cu_seqlens_k = ToAtenTensor<kDev>(
        const_cast<void*>(cu_seqlens_k.data()), cu_seqlens_k_shape_,
        cu_seqlens_k_strides_, cu_seqlens_k_dtype_, device_index_);
    auto at_out = ToAtenTensor<kDev>(out.data(), out_shape_, out_strides_,
                                     out_dtype_, device_index_);
    std::optional<at::Tensor> at_softmax_lse;
    std::optional<at::Tensor> at_s_dmask;
    if (softmax_lse.has_value()) {
      at_softmax_lse.emplace(ToAtenTensor<kDev>(
          softmax_lse->data(), softmax_lse_shape_, softmax_lse_strides_,
          softmax_lse_dtype_, device_index_));
      at_s_dmask.emplace(ToAtenTensor<kDev>(s_dmask->data(), s_dmask_shape_,
                                            s_dmask_strides_, s_dmask_dtype_,
                                            device_index_));
    }

    const auto options = Policy::MakeOptions(causal, window_size,
                                             at_cu_seqlens_q, at_cu_seqlens_k);

    auto result = at::_flash_attention_forward(
        at_q, at_k, at_v, at_cu_seqlens_q, at_cu_seqlens_k, max_seqlen_q,
        max_seqlen_k, dropout_p, causal, false, softmax_scale,
        options.window_size_left, options.window_size_right, std::nullopt,
        std::nullopt);

    // ATen owns the returned tensors. Preserve the InfiniOps trailing-output
    // ABI by copying them into caller-provided buffers on the selected stream.
    at_out.copy_(std::get<0>(result));
    if (at_softmax_lse.has_value()) {
      const auto& result_softmax_lse = std::get<1>(result);
      if (result_softmax_lse.dim() == 3) {
        const auto batch_size = result_softmax_lse.size(0);
        const auto q_lengths = at_cu_seqlens_q.narrow(0, 1, batch_size) -
                               at_cu_seqlens_q.narrow(0, 0, batch_size);
        const auto positions =
            at::arange(result_softmax_lse.size(2), at_cu_seqlens_q.options());
        const auto valid_positions =
            positions.unsqueeze(0).lt(q_lengths.unsqueeze(1));
        const auto packed_softmax_lse =
            result_softmax_lse.transpose(1, 2)
                .masked_select(valid_positions.unsqueeze(2))
                .view({at_q.size(0), at_q.size(1)})
                .transpose(0, 1);
        at_softmax_lse->copy_(packed_softmax_lse);
      } else {
        at_softmax_lse->copy_(result_softmax_lse);
      }
      at_s_dmask->copy_(std::get<4>(result));
    }
  };

  using Backend = C10<kDev>;
  const typename Backend::StreamGuard stream_guard{
      Backend::GetStreamFromExternal(stream_, device_index_)};
  run();
}

}  // namespace infini::ops

#endif  // INFINI_OPS_TORCH_FLASH_ATTN_VARLEN_FUNC_ATEN_IMPL_H_
