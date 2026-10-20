#ifndef INFINI_OPS_NVIDIA_GET_CUTLASS_MOE_MM_DATA_KERNEL_H_
#define INFINI_OPS_NVIDIA_GET_CUTLASS_MOE_MM_DATA_KERNEL_H_

#include <optional>

#include "base/get_cutlass_moe_mm_data.h"

namespace infini::ops {

template <>
class Operator<GetCutlassMoeMmData, Device::Type::kNvidia, 0>
    : public GetCutlassMoeMmData {
 public:
  using GetCutlassMoeMmData::GetCutlassMoeMmData;

  using GetCutlassMoeMmData::operator();

  void operator()(const Tensor topk_ids, const int64_t num_experts,
                  const int64_t n, const int64_t k, const bool is_gated,
                  Tensor expert_offsets, Tensor problem_sizes1,
                  Tensor problem_sizes2, Tensor input_permutation,
                  Tensor output_permutation,
                  std::optional<Tensor> blockscale_offsets) const override;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_NVIDIA_GET_CUTLASS_MOE_MM_DATA_KERNEL_H_
