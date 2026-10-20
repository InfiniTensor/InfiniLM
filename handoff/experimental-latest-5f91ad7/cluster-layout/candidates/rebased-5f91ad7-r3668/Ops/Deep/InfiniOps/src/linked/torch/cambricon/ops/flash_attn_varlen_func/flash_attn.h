#ifndef INFINI_OPS_LINKED_TORCH_CAMBRICON_OPS_FLASH_ATTN_VARLEN_FUNC_FLASH_ATTN_H_
#define INFINI_OPS_LINKED_TORCH_CAMBRICON_OPS_FLASH_ATTN_VARLEN_FUNC_FLASH_ATTN_H_

#include "base/flash_attn_varlen_func.h"

namespace infini::ops {

template <>
class Operator<FlashAttnVarlenFunc, Device::Type::kCambricon, 16>
    : public FlashAttnVarlenFunc {
 public:
  using FlashAttnVarlenFunc::FlashAttnVarlenFunc;
  using FlashAttnVarlenFunc::operator();

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
                  std::optional<Tensor> s_dmask) const override;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_LINKED_TORCH_CAMBRICON_OPS_FLASH_ATTN_VARLEN_FUNC_FLASH_ATTN_H_
