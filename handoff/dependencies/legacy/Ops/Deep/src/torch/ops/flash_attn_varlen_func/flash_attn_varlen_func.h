#ifndef INFINI_OPS_TORCH_FLASH_ATTN_VARLEN_FUNC_H_
#define INFINI_OPS_TORCH_FLASH_ATTN_VARLEN_FUNC_H_

#include "base/flash_attn_varlen_func.h"

namespace infini::ops {

template <Device::Type kDev>
class AtenFlashAttnVarlenFunc : public FlashAttnVarlenFunc {
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

template <>
class Operator<FlashAttnVarlenFunc, Device::Type::kNvidia, 8>
    : public AtenFlashAttnVarlenFunc<Device::Type::kNvidia> {
 public:
  using Base = AtenFlashAttnVarlenFunc<Device::Type::kNvidia>;
  using Base::Base;
  using Base::operator();
};

template <>
class Operator<FlashAttnVarlenFunc, Device::Type::kMoore, 8>
    : public AtenFlashAttnVarlenFunc<Device::Type::kMoore> {
 public:
  using Base = AtenFlashAttnVarlenFunc<Device::Type::kMoore>;
  using Base::Base;
  using Base::operator();

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

#endif  // INFINI_OPS_TORCH_FLASH_ATTN_VARLEN_FUNC_H_
