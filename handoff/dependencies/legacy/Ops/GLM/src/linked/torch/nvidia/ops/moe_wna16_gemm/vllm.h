#ifndef INFINI_OPS_LINKED_TORCH_NVIDIA_OPS_MOE_WNA16_GEMM_VLLM_H_
#define INFINI_OPS_LINKED_TORCH_NVIDIA_OPS_MOE_WNA16_GEMM_VLLM_H_

#include <optional>

#include "linked/torch/ops/moe_wna16_gemm.h"
#include "torch/nvidia/c10.h"

namespace infini::ops::linked::torch::nvidia {

struct VllmMoeWna16Gemm : C10<Device::Type::kNvidia> {
  static void Call(at::Tensor input, at::Tensor output, at::Tensor b_qweight,
                   at::Tensor b_scales, std::optional<at::Tensor> b_qzeros,
                   std::optional<at::Tensor> topk_weights,
                   at::Tensor sorted_token_ids, at::Tensor expert_ids,
                   at::Tensor num_tokens_post_pad, int64_t top_k,
                   int64_t block_size_m, int64_t block_size_n,
                   int64_t block_size_k, int64_t bit);
};

}  // namespace infini::ops::linked::torch::nvidia

namespace infini::ops::linked::torch {

extern template class TorchMoeWna16Gemm<
    ::infini::ops::linked::torch::nvidia::VllmMoeWna16Gemm>;

}  // namespace infini::ops::linked::torch

namespace infini::ops {

template <>
class Operator<MoeWna16Gemm, Device::Type::kNvidia, 16>
    : public linked::torch::TorchMoeWna16Gemm<
          linked::torch::nvidia::VllmMoeWna16Gemm> {
 public:
  using linked::torch::TorchMoeWna16Gemm<
      linked::torch::nvidia::VllmMoeWna16Gemm>::TorchMoeWna16Gemm;

  using linked::torch::TorchMoeWna16Gemm<
      linked::torch::nvidia::VllmMoeWna16Gemm>::operator();
};

}  // namespace infini::ops

#endif  // INFINI_OPS_LINKED_TORCH_NVIDIA_OPS_MOE_WNA16_GEMM_VLLM_H_
