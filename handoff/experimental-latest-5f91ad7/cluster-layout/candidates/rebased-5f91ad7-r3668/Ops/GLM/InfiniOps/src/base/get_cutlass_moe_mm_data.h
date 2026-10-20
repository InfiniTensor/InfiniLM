#ifndef INFINI_OPS_BASE_GET_CUTLASS_MOE_MM_DATA_H_
#define INFINI_OPS_BASE_GET_CUTLASS_MOE_MM_DATA_H_

#include <cassert>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>

#include "operator.h"

namespace infini::ops {

// Aligned with vLLM `get_cutlass_moe_mm_data`.
class GetCutlassMoeMmData : public Operator<GetCutlassMoeMmData> {
 public:
  GetCutlassMoeMmData(const Tensor topk_ids, const int64_t num_experts,
                      const int64_t n, const int64_t k, Tensor expert_offsets,
                      Tensor problem_sizes1, Tensor problem_sizes2,
                      Tensor input_permutation, Tensor output_permutation,
                      std::optional<Tensor> blockscale_offsets = std::nullopt)
      : GetCutlassMoeMmData{topk_ids,
                            num_experts,
                            n,
                            k,
                            true,
                            expert_offsets,
                            problem_sizes1,
                            problem_sizes2,
                            input_permutation,
                            output_permutation,
                            blockscale_offsets} {}

  GetCutlassMoeMmData(const Tensor topk_ids, const int64_t num_experts,
                      const int64_t n, const int64_t k, const bool is_gated,
                      Tensor expert_offsets, Tensor problem_sizes1,
                      Tensor problem_sizes2, Tensor input_permutation,
                      Tensor output_permutation,
                      std::optional<Tensor> blockscale_offsets = std::nullopt)
      : topk_ids_metadata_{topk_ids},
        expert_offsets_metadata_{expert_offsets},
        problem_sizes1_metadata_{problem_sizes1},
        problem_sizes2_metadata_{problem_sizes2},
        input_permutation_metadata_{input_permutation},
        output_permutation_metadata_{output_permutation},
        blockscale_offsets_metadata_{blockscale_offsets},
        num_experts_{num_experts},
        n_{n},
        k_{k},
        is_gated_{is_gated},
        numel_{topk_ids.numel()},
        topk_{topk_ids.ndim() == 2 ? topk_ids.size(1) : 0},
        device_index_{topk_ids.device().index()} {
    Validate(topk_ids, expert_offsets, problem_sizes1, problem_sizes2,
             input_permutation, output_permutation, blockscale_offsets);
  }

  void operator()(
      const Tensor topk_ids, const int64_t num_experts, const int64_t n,
      const int64_t k, Tensor expert_offsets, Tensor problem_sizes1,
      Tensor problem_sizes2, Tensor input_permutation,
      Tensor output_permutation,
      std::optional<Tensor> blockscale_offsets = std::nullopt) const {
    (*this)(topk_ids, num_experts, n, k, true, expert_offsets, problem_sizes1,
            problem_sizes2, input_permutation, output_permutation,
            blockscale_offsets);
  }

  virtual void operator()(
      const Tensor topk_ids, const int64_t num_experts, const int64_t n,
      const int64_t k, const bool is_gated, Tensor expert_offsets,
      Tensor problem_sizes1, Tensor problem_sizes2, Tensor input_permutation,
      Tensor output_permutation,
      std::optional<Tensor> blockscale_offsets = std::nullopt) const = 0;

 protected:
  void ValidateCallMetadata(
      const Tensor topk_ids, const int64_t num_experts, const int64_t n,
      const int64_t k, const bool is_gated, const Tensor expert_offsets,
      const Tensor problem_sizes1, const Tensor problem_sizes2,
      const Tensor input_permutation, const Tensor output_permutation,
      const std::optional<Tensor> blockscale_offsets) const {
    assert(num_experts == num_experts_ && n == n_ && k == k_ &&
           is_gated == is_gated_ &&
           "`GetCutlassMoeMmData` attributes changed after descriptor "
           "creation");
    assert(CallMetadataMatches(topk_ids, expert_offsets, problem_sizes1,
                               problem_sizes2, input_permutation,
                               output_permutation, blockscale_offsets) &&
           "`GetCutlassMoeMmData` tensor metadata differs from its descriptor");
  }

  Tensor topk_ids_metadata_;

  Tensor expert_offsets_metadata_;

  Tensor problem_sizes1_metadata_;

  Tensor problem_sizes2_metadata_;

  Tensor input_permutation_metadata_;

  Tensor output_permutation_metadata_;

  std::optional<Tensor> blockscale_offsets_metadata_;

  int64_t num_experts_{0};

  int64_t n_{0};

  int64_t k_{0};

  bool is_gated_{true};

  Tensor::Size numel_{0};

  Tensor::Size topk_{0};

  int device_index_{0};

 private:
  bool CallMetadataMatches(
      const Tensor topk_ids, const Tensor expert_offsets,
      const Tensor problem_sizes1, const Tensor problem_sizes2,
      const Tensor input_permutation, const Tensor output_permutation,
      const std::optional<Tensor> blockscale_offsets) const {
    const std::equal_to<Tensor> same_metadata;
    const auto same_blockscale_metadata =
        blockscale_offsets_metadata_.has_value() ==
            blockscale_offsets.has_value() &&
        (!blockscale_offsets_metadata_ ||
         same_metadata(*blockscale_offsets_metadata_, *blockscale_offsets));

    return same_metadata(topk_ids_metadata_, topk_ids) &&
           same_metadata(expert_offsets_metadata_, expert_offsets) &&
           same_metadata(problem_sizes1_metadata_, problem_sizes1) &&
           same_metadata(problem_sizes2_metadata_, problem_sizes2) &&
           same_metadata(input_permutation_metadata_, input_permutation) &&
           same_metadata(output_permutation_metadata_, output_permutation) &&
           same_blockscale_metadata;
  }

  void Validate(const Tensor topk_ids, const Tensor expert_offsets,
                const Tensor problem_sizes1, const Tensor problem_sizes2,
                const Tensor input_permutation, const Tensor output_permutation,
                const std::optional<Tensor> blockscale_offsets) const {
    assert(topk_ids.ndim() == 2 &&
           "`GetCutlassMoeMmData` requires 2D `topk_ids`");
    assert(topk_ids.dtype() == DataType::kInt32 &&
           "`GetCutlassMoeMmData` requires int32 `topk_ids`");
    assert(topk_ids.IsContiguous() &&
           "`GetCutlassMoeMmData` requires contiguous `topk_ids`");
    assert(numel_ > 0 && topk_ > 0 &&
           "`GetCutlassMoeMmData` requires non-empty `topk_ids`");
    assert(num_experts_ > 0 &&
           num_experts_ <= std::numeric_limits<int32_t>::max() / 3 &&
           "`GetCutlassMoeMmData` requires indexable int32 `num_experts`");
    const auto max_n = is_gated_ ? std::numeric_limits<int32_t>::max() / 2
                                 : std::numeric_limits<int32_t>::max();
    assert(n_ > 0 && n_ <= max_n && k_ > 0 &&
           k_ <= std::numeric_limits<int32_t>::max() &&
           "`GetCutlassMoeMmData` requires positive int32 GEMM dimensions");
    assert(numel_ <=
               static_cast<Tensor::Size>(std::numeric_limits<int32_t>::max()) &&
           "`GetCutlassMoeMmData` requires int32-addressable routing indices");
    if (blockscale_offsets) {
      constexpr int64_t kBlockscalePadding = 127;
      const int64_t max_blockscale_padding = num_experts_ * kBlockscalePadding;
      assert(max_blockscale_padding <= std::numeric_limits<int32_t>::max() &&
             static_cast<int64_t>(numel_) <=
                 std::numeric_limits<int32_t>::max() - max_blockscale_padding &&
             "`GetCutlassMoeMmData` blockscale offsets must fit int32");
    }

    const auto same_device_as_topk_ids = [&](const Tensor tensor) {
      return tensor.device().type() == topk_ids.device().type() &&
             tensor.device().index() == topk_ids.device().index();
    };
    assert(
        same_device_as_topk_ids(expert_offsets) &&
        same_device_as_topk_ids(problem_sizes1) &&
        same_device_as_topk_ids(problem_sizes2) &&
        same_device_as_topk_ids(input_permutation) &&
        same_device_as_topk_ids(output_permutation) &&
        (!blockscale_offsets || same_device_as_topk_ids(*blockscale_offsets)) &&
        "`GetCutlassMoeMmData` requires all tensors on the same device");

    const auto num_experts = static_cast<Tensor::Size>(num_experts_);
    assert(expert_offsets.ndim() == 1 &&
           expert_offsets.numel() == num_experts + 1 &&
           "`GetCutlassMoeMmData` requires `expert_offsets` shape "
           "[`num_experts + 1`]");
    assert(problem_sizes1.ndim() == 2 &&
           problem_sizes1.size(0) == num_experts &&
           problem_sizes1.size(1) == 3 && problem_sizes2.ndim() == 2 &&
           problem_sizes2.shape() == problem_sizes1.shape() &&
           "`GetCutlassMoeMmData` requires problem sizes shape "
           "[`num_experts`, 3]");
    assert(input_permutation.ndim() == 1 &&
           input_permutation.numel() == numel_ &&
           output_permutation.ndim() == 1 &&
           output_permutation.numel() == numel_ &&
           "`GetCutlassMoeMmData` requires permutation shape "
           "[`topk_ids.numel()`]");
    assert((!blockscale_offsets ||
            (blockscale_offsets->ndim() == 1 &&
             blockscale_offsets->numel() == num_experts + 1)) &&
           "`GetCutlassMoeMmData` requires `blockscale_offsets` shape "
           "[`num_experts + 1`]");

    const auto is_int32_contiguous = [](const Tensor tensor) {
      return tensor.dtype() == DataType::kInt32 && tensor.IsContiguous();
    };
    assert(is_int32_contiguous(expert_offsets) &&
           is_int32_contiguous(problem_sizes1) &&
           is_int32_contiguous(problem_sizes2) &&
           is_int32_contiguous(input_permutation) &&
           is_int32_contiguous(output_permutation) &&
           (!blockscale_offsets || is_int32_contiguous(*blockscale_offsets)) &&
           "`GetCutlassMoeMmData` requires contiguous int32 outputs");
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_GET_CUTLASS_MOE_MM_DATA_H_
