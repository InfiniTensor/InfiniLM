#include "linked/torch/hygon/ops/flash_attn_varlen_func/flash_attn.h"

#include <cassert>

#include "linked/torch/ops/flash_attn_varlen_func.h"
#include "torch/hygon/c10.h"

extern "C" std::vector<at::Tensor> mha_varlen_fwd(
    at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    std::optional<at::Tensor>& out, const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k, std::optional<at::Tensor>& seqused_k,
    std::optional<const at::Tensor>& leftpad_k,
    std::optional<at::Tensor>& block_table,
    std::optional<at::Tensor>& alibi_slopes, int max_seqlen_q, int max_seqlen_k,
    float dropout_p, float softmax_scale, bool zero_tensors, bool causal,
    int window_size_left, int window_size_right, float softcap,
    bool return_softmax, std::optional<at::Tensor> q_descale,
    std::optional<at::Tensor> k_descale, std::optional<at::Tensor> v_descale,
    std::optional<at::Generator> generator,
    const std::optional<at::Tensor>& s_aux);

namespace infini::ops::linked::torch::hygon {

struct FlashAttnVarlen : C10<Device::Type::kHygon> {
  static std::vector<at::Tensor> Call(
      at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
      std::optional<at::Tensor>& out, const at::Tensor& cu_seqlens_q,
      const at::Tensor& cu_seqlens_k, std::optional<at::Tensor>& seqused_k,
      std::optional<const at::Tensor>& leftpad_k,
      std::optional<at::Tensor>& block_table,
      std::optional<at::Tensor>& alibi_slopes, int max_seqlen_q,
      int max_seqlen_k, float dropout_p, float softmax_scale, bool zero_tensors,
      bool causal, int window_size_left, int window_size_right, float softcap,
      bool return_softmax, std::optional<at::Generator> generator);
};

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
  assert(!alibi_slopes.has_value() &&
         "Hygon FlashAttention varlen does not support ALiBi.");

  auto q_work = q.contiguous();
  auto k_work = k.contiguous();
  auto v_work = v.contiguous();

  if (block_table.has_value() && k.dim() == 4 && v.dim() == 4) {
    constexpr int64_t kPageSize = 64;
    const int64_t num_blocks = k.size(0);
    const int64_t block_size = k.size(1);
    const int64_t num_kv_heads = k.size(2);
    const int64_t head_dim = k.size(3);
    assert(block_size % kPageSize == 0 &&
           "Hygon FlashAttention requires paged KV block size to be divisible "
           "by 64.");

    const int64_t pages_per_block = block_size / kPageSize;
    k_work = k_work
                 .reshape({num_blocks, pages_per_block, kPageSize, num_kv_heads,
                           head_dim})
                 .reshape({num_blocks * pages_per_block, kPageSize,
                           num_kv_heads, head_dim})
                 .contiguous();
    v_work = v_work
                 .reshape({num_blocks, pages_per_block, kPageSize, num_kv_heads,
                           head_dim})
                 .reshape({num_blocks * pages_per_block, kPageSize,
                           num_kv_heads, head_dim})
                 .contiguous();
    if (pages_per_block != 1) {
      auto offsets = at::arange(pages_per_block, block_table->options())
                         .view({1, 1, pages_per_block});
      block_table = (block_table->unsqueeze(-1) * pages_per_block + offsets)
                        .reshape({block_table->size(0),
                                  block_table->size(1) * pages_per_block})
                        .contiguous();
    }
  }

  std::optional<at::Tensor> q_descale;
  std::optional<at::Tensor> k_descale;
  std::optional<at::Tensor> v_descale;
  std::optional<at::Tensor> s_aux;
  return ::mha_varlen_fwd(q_work, k_work, v_work, out, cu_seqlens_q,
                          cu_seqlens_k, seqused_k, leftpad_k, block_table,
                          alibi_slopes, max_seqlen_q, max_seqlen_k, dropout_p,
                          softmax_scale, zero_tensors, causal, window_size_left,
                          window_size_right, softcap, return_softmax, q_descale,
                          k_descale, v_descale, generator, s_aux);
}

}  // namespace infini::ops::linked::torch::hygon

namespace infini::ops::linked::torch {

template class TorchFlashAttnVarlenFunc<
    ::infini::ops::linked::torch::hygon::FlashAttnVarlen>;

}  // namespace infini::ops::linked::torch

namespace infini::ops {

void Operator<FlashAttnVarlenFunc, Device::Type::kHygon, 16>::operator()(
    const Tensor q, const Tensor k, const Tensor v, const Tensor cu_seqlens_q,
    const Tensor cu_seqlens_k, const std::optional<Tensor> alibi_slopes,
    const std::optional<Tensor> block_table, const int64_t max_seqlen_q,
    const int64_t max_seqlen_k, const double dropout_p,
    const std::optional<double> softmax_scale, const bool causal,
    const std::vector<int64_t> window_size, const double softcap,
    const bool deterministic, const bool return_attn_probs, Tensor out,
    std::optional<Tensor> softmax_lse, std::optional<Tensor> s_dmask) const {
  using Delegate = linked::torch::TorchFlashAttnVarlenFunc<
      linked::torch::hygon::FlashAttnVarlen>;
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
