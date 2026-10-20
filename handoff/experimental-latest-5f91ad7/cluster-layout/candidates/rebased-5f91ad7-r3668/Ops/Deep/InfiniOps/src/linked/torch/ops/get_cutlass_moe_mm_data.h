#ifndef INFINI_OPS_LINKED_TORCH_OPS_GET_CUTLASS_MOE_MM_DATA_H_
#define INFINI_OPS_LINKED_TORCH_OPS_GET_CUTLASS_MOE_MM_DATA_H_

#include <optional>
#include <utility>

#include "base/get_cutlass_moe_mm_data.h"
#include "torch/tensor_.h"

namespace infini::ops::linked::torch {

template <typename Backend>
class TorchGetCutlassMoeMmData : public ::infini::ops::GetCutlassMoeMmData {
 public:
  using ::infini::ops::GetCutlassMoeMmData::GetCutlassMoeMmData;

  using ::infini::ops::GetCutlassMoeMmData::operator();

  void operator()(const Tensor topk_ids, const int64_t num_experts,
                  const int64_t n, const int64_t k, const bool is_gated,
                  Tensor expert_offsets, Tensor problem_sizes1,
                  Tensor problem_sizes2, Tensor input_permutation,
                  Tensor output_permutation,
                  std::optional<Tensor> blockscale_offsets) const override {
    ValidateCallMetadata(topk_ids, num_experts, n, k, is_gated, expert_offsets,
                         problem_sizes1, problem_sizes2, input_permutation,
                         output_permutation, blockscale_offsets);
    TORCH_CHECK(is_gated,
                "The linked vLLM `get_cutlass_moe_mm_data` implementation only "
                "supports `is_gated=true`.");
    Backend::ValidateDevice(device_index_);

    const typename Backend::StreamGuard stream_guard{
        Backend::GetStreamFromExternal(stream_, device_index_)};
    auto at_topk_ids = ToAtenTensor<Backend::kDeviceType>(
        const_cast<void*>(topk_ids.data()), topk_ids_metadata_.shape(),
        topk_ids_metadata_.strides(), topk_ids_metadata_.dtype(),
        device_index_);
    auto at_expert_offsets = ToAtenTensor<Backend::kDeviceType>(
        expert_offsets.data(), expert_offsets_metadata_.shape(),
        expert_offsets_metadata_.strides(), expert_offsets_metadata_.dtype(),
        device_index_);
    auto at_problem_sizes1 = ToAtenTensor<Backend::kDeviceType>(
        problem_sizes1.data(), problem_sizes1_metadata_.shape(),
        problem_sizes1_metadata_.strides(), problem_sizes1_metadata_.dtype(),
        device_index_);
    auto at_problem_sizes2 = ToAtenTensor<Backend::kDeviceType>(
        problem_sizes2.data(), problem_sizes2_metadata_.shape(),
        problem_sizes2_metadata_.strides(), problem_sizes2_metadata_.dtype(),
        device_index_);
    auto at_input_permutation = ToAtenTensor<Backend::kDeviceType>(
        input_permutation.data(), input_permutation_metadata_.shape(),
        input_permutation_metadata_.strides(),
        input_permutation_metadata_.dtype(), device_index_);
    auto at_output_permutation = ToAtenTensor<Backend::kDeviceType>(
        output_permutation.data(), output_permutation_metadata_.shape(),
        output_permutation_metadata_.strides(),
        output_permutation_metadata_.dtype(), device_index_);
    std::optional<at::Tensor> at_blockscale_offsets;
    if (blockscale_offsets.has_value()) {
      const auto& metadata = *blockscale_offsets_metadata_;
      at_blockscale_offsets.emplace(ToAtenTensor<Backend::kDeviceType>(
          blockscale_offsets->data(), metadata.shape(), metadata.strides(),
          metadata.dtype(), device_index_));
    }

    Backend::Call(std::move(at_topk_ids), std::move(at_expert_offsets),
                  std::move(at_problem_sizes1), std::move(at_problem_sizes2),
                  std::move(at_input_permutation),
                  std::move(at_output_permutation), num_experts, n, k,
                  std::move(at_blockscale_offsets));
  }
};

}  // namespace infini::ops::linked::torch

#endif  // INFINI_OPS_LINKED_TORCH_OPS_GET_CUTLASS_MOE_MM_DATA_H_
