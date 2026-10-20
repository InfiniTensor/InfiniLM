#pragma once

#include <memory>

#include "base/flash_attn_varlen_func.h"

#define INFINI_NATIVE_VARLEN_ARGS                                            \
  const Tensor q, const Tensor k, const Tensor v, const Tensor cu_seqlens_q, \
      const Tensor cu_seqlens_k, const std::optional<Tensor> alibi_slopes,   \
      const std::optional<Tensor> block_table, const int64_t max_seqlen_q,   \
      const int64_t max_seqlen_k, const double dropout_p,                    \
      const std::optional<double> softmax_scale, const bool causal,          \
      const std::vector<int64_t> window_size, const double softcap,          \
      const bool deterministic, const bool return_attn_probs, Tensor out,    \
      std::optional<Tensor> softmax_lse, std::optional<Tensor> s_dmask
#define INFINI_NATIVE_VARLEN_VALUES                                 \
  q, k, v, cu_seqlens_q, cu_seqlens_k, alibi_slopes, block_table,   \
      max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, causal, \
      window_size, softcap, deterministic, return_attn_probs, out,  \
      softmax_lse, s_dmask

namespace infini::ops {

// Inference-only rounded widths and vector layouts are checked by supports().
// Automatic selection leaves other configurations to an existing provider;
// construction failures propagate.
template <>
class Operator<FlashAttnVarlenFunc, Device::Type::kNvidia, 0>
    : public FlashAttnVarlenFunc {
 public:
  static bool supports(INFINI_NATIVE_VARLEN_ARGS);

  Operator(INFINI_NATIVE_VARLEN_ARGS)
      : FlashAttnVarlenFunc(
            checked_q(INFINI_NATIVE_VARLEN_VALUES), k, v, cu_seqlens_q,
            cu_seqlens_k, alibi_slopes, block_table, max_seqlen_q, max_seqlen_k,
            dropout_p, softmax_scale, causal, window_size, softcap,
            deterministic, return_attn_probs, out, softmax_lse, s_dmask) {
    prepare(INFINI_NATIVE_VARLEN_VALUES);
  }
  ~Operator() override;
  using FlashAttnVarlenFunc::operator();

  void operator()(INFINI_NATIVE_VARLEN_ARGS) const override;

  // Graph callers must supply stable external workspace for the graph lifetime.
  bool capture_safe() const override { return true; }
  std::size_t workspace_size_in_bytes() const override;

 private:
  static Tensor checked_q(INFINI_NATIVE_VARLEN_ARGS) {
    if (!supports(INFINI_NATIVE_VARLEN_VALUES))
      throw std::invalid_argument(
          "unsupported native varlen Attention configuration");
    return q;
  }
  void prepare(INFINI_NATIVE_VARLEN_ARGS);
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace infini::ops

#undef INFINI_NATIVE_VARLEN_ARGS
#undef INFINI_NATIVE_VARLEN_VALUES
