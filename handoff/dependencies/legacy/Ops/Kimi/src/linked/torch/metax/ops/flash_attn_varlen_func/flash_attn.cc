#include "linked/torch/metax/ops/flash_attn_varlen_func/flash_attn.h"

#include <ATen/core/Generator.h>

#include "linked/torch/ops/flash_attn_varlen_func.h"
#include "torch/metax/c10.h"

std::vector<at::Tensor> mha_varlen_fwd(
    at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    std::optional<at::Tensor>& out, const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k, std::optional<at::Tensor>& seqused_k,
    std::optional<const at::Tensor>& leftpad_k,
    std::optional<at::Tensor>& block_table,
    std::optional<at::Tensor>& alibi_slopes, int max_seqlen_q, int max_seqlen_k,
    float dropout_p, float softmax_scale, bool zero_tensors, bool causal,
    int window_size_left, int window_size_right, float softcap,
    bool return_softmax, std::optional<at::Generator> generator,
    std::optional<at::Tensor>& flash_attn_mars_ext);

namespace infini::ops::linked::torch::metax {

struct FlashAttnVarlen : C10<Device::Type::kMetax> {
  static std::vector<at::Tensor> Call(
      at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
      std::optional<at::Tensor>& out, const at::Tensor& cu_seqlens_q,
      const at::Tensor& cu_seqlens_k, std::optional<at::Tensor>& seqused_k,
      std::optional<const at::Tensor>& leftpad_k,
      std::optional<at::Tensor>& block_table,
      std::optional<at::Tensor>& alibi_slopes, int max_seqlen_q,
      int max_seqlen_k, float dropout_p, float softmax_scale, bool zero_tensors,
      bool causal, int window_size_left, int window_size_right, float softcap,
      bool return_softmax, std::optional<at::Generator> generator) {
    std::optional<at::Tensor> flash_attn_mars_ext;
    return ::mha_varlen_fwd(q, k, v, out, cu_seqlens_q, cu_seqlens_k, seqused_k,
                            leftpad_k, block_table, alibi_slopes, max_seqlen_q,
                            max_seqlen_k, dropout_p, softmax_scale,
                            zero_tensors, causal, window_size_left,
                            window_size_right, softcap, return_softmax,
                            generator, flash_attn_mars_ext);
  }
};

}  // namespace infini::ops::linked::torch::metax

namespace infini::ops {

void Operator<FlashAttnVarlenFunc, Device::Type::kMetax, 16>::operator()(
    const Tensor q, const Tensor k, const Tensor v, const Tensor cu_seqlens_q,
    const Tensor cu_seqlens_k, const std::optional<Tensor> alibi_slopes,
    const std::optional<Tensor> block_table, const int64_t max_seqlen_q,
    const int64_t max_seqlen_k, const double dropout_p,
    const std::optional<double> softmax_scale, const bool causal,
    const std::vector<int64_t> window_size, const double softcap,
    const bool deterministic, const bool return_attn_probs, Tensor out,
    std::optional<Tensor> softmax_lse, std::optional<Tensor> s_dmask) const {
  using Delegate = linked::torch::TorchFlashAttnVarlenFunc<
      linked::torch::metax::FlashAttnVarlen>;
  if (!delegate_) {
    delegate_ = std::make_unique<Delegate>(
        q, k, v, cu_seqlens_q, cu_seqlens_k, alibi_slopes, block_table,
        max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, causal,
        window_size, softcap, deterministic, return_attn_probs, out,
        softmax_lse, s_dmask);
  }
  delegate_->set_stream(stream_);
  (*delegate_)(q, k, v, cu_seqlens_q, cu_seqlens_k, alibi_slopes, block_table,
               max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, causal,
               window_size, softcap, deterministic, return_attn_probs, out,
               softmax_lse, s_dmask);
}

}  // namespace infini::ops
