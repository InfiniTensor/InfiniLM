#ifndef INFINI_OPS_NVIDIA_MOE_WNA16_GEMM_KERNEL_H_
#define INFINI_OPS_NVIDIA_MOE_WNA16_GEMM_KERNEL_H_

#include <optional>

#include "base/moe_wna16_gemm.h"

namespace infini::ops {

template <>
class Operator<MoeWna16Gemm, Device::Type::kNvidia, 0> : public MoeWna16Gemm {
 public:
  Operator(const Tensor input, const Tensor b_qweight, const Tensor b_scales,
           std::optional<Tensor> b_qzeros, std::optional<Tensor> topk_weights,
           const Tensor sorted_token_ids, const Tensor expert_ids,
           const Tensor num_tokens_post_pad, const int64_t top_k,
           const int64_t block_size_m, const int64_t block_size_n,
           const int64_t block_size_k, const int64_t bit, Tensor output);

  void operator()(const Tensor input, const Tensor b_qweight,
                  const Tensor b_scales, std::optional<Tensor> b_qzeros,
                  std::optional<Tensor> topk_weights,
                  const Tensor sorted_token_ids, const Tensor expert_ids,
                  const Tensor num_tokens_post_pad, const int64_t top_k,
                  const int64_t block_size_m, const int64_t block_size_n,
                  const int64_t block_size_k, const int64_t bit,
                  Tensor output) const override;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_NVIDIA_MOE_WNA16_GEMM_KERNEL_H_
