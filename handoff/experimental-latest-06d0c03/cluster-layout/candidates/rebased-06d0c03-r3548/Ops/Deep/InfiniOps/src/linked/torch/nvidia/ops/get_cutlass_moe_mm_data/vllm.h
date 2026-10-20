#ifndef INFINI_OPS_LINKED_TORCH_NVIDIA_OPS_GET_CUTLASS_MOE_MM_DATA_VLLM_H_
#define INFINI_OPS_LINKED_TORCH_NVIDIA_OPS_GET_CUTLASS_MOE_MM_DATA_VLLM_H_

#include "linked/torch/ops/get_cutlass_moe_mm_data.h"
#include "torch/nvidia/c10.h"

namespace infini::ops::linked::torch::nvidia {

struct VllmGetCutlassMoeMmData : C10<Device::Type::kNvidia> {
  static void ValidateDevice(int device_index);

  static void Call(at::Tensor topk_ids, at::Tensor expert_offsets,
                   at::Tensor problem_sizes1, at::Tensor problem_sizes2,
                   at::Tensor input_permutation, at::Tensor output_permutation,
                   int64_t num_experts, int64_t n, int64_t k,
                   std::optional<at::Tensor> blockscale_offsets);
};

}  // namespace infini::ops::linked::torch::nvidia

namespace infini::ops::linked::torch {

extern template class TorchGetCutlassMoeMmData<
    ::infini::ops::linked::torch::nvidia::VllmGetCutlassMoeMmData>;

}  // namespace infini::ops::linked::torch

namespace infini::ops {

template <>
class Operator<GetCutlassMoeMmData, Device::Type::kNvidia, 16>
    : public linked::torch::TorchGetCutlassMoeMmData<
          linked::torch::nvidia::VllmGetCutlassMoeMmData> {
 public:
  using linked::torch::TorchGetCutlassMoeMmData<
      linked::torch::nvidia::VllmGetCutlassMoeMmData>::TorchGetCutlassMoeMmData;

  using linked::torch::TorchGetCutlassMoeMmData<
      linked::torch::nvidia::VllmGetCutlassMoeMmData>::operator();
};

}  // namespace infini::ops

#endif  // INFINI_OPS_LINKED_TORCH_NVIDIA_OPS_GET_CUTLASS_MOE_MM_DATA_VLLM_H_
