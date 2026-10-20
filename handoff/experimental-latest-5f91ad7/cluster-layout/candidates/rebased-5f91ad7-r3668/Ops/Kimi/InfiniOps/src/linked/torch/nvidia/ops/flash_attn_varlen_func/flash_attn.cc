#include "linked/torch/nvidia/ops/flash_attn_varlen_func/flash_attn.h"

namespace flash {

std::vector<at::Tensor> mha_varlen_fwd(
    at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    std::optional<at::Tensor>& out, const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k, std::optional<at::Tensor>& seqused_k,
    std::optional<const at::Tensor>& leftpad_k,
    std::optional<at::Tensor>& block_table,
    std::optional<at::Tensor>& alibi_slopes, int max_seqlen_q, int max_seqlen_k,
    float dropout_p, float softmax_scale, bool zero_tensors, bool causal,
    int window_size_left, int window_size_right, float softcap,
    bool return_softmax, std::optional<at::Generator> generator);

}  // namespace flash

namespace infini::ops::linked::torch::nvidia {

std::vector<at::Tensor> FlashAttnVarlen::Call(
    at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    std::optional<at::Tensor>& out, const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k, std::optional<at::Tensor>& seqused_k,
    std::optional<const at::Tensor>& leftpad_k,
    std::optional<at::Tensor>& block_table,
    std::optional<at::Tensor>& alibi_slopes, int max_seqlen_q, int max_seqlen_k,
    float dropout_p, float softmax_scale, bool zero_tensors, bool causal,
    int window_size_left, int window_size_right, float softcap,
    bool return_softmax, std::optional<at::Generator> generator) {
  return flash::mha_varlen_fwd(
      q, k, v, out, cu_seqlens_q, cu_seqlens_k, seqused_k, leftpad_k,
      block_table, alibi_slopes, max_seqlen_q, max_seqlen_k, dropout_p,
      softmax_scale, zero_tensors, causal, window_size_left, window_size_right,
      softcap, return_softmax, generator);
}

}  // namespace infini::ops::linked::torch::nvidia

namespace infini::ops::linked::torch {

template class TorchFlashAttnVarlenFunc<
    ::infini::ops::linked::torch::nvidia::FlashAttnVarlen>;

}  // namespace infini::ops::linked::torch
