#ifndef INFINI_OPS_BASE_GROUPED_TOPK_H_
#define INFINI_OPS_BASE_GROUPED_TOPK_H_

#include <cassert>
#include <cstdint>
#include <functional>
#include <limits>

#include "operator.h"

namespace infini::ops {

// Aligned with vLLM's low-level `_moe_C::grouped_topk` operator.
class GroupedTopk : public Operator<GroupedTopk> {
 public:
  GroupedTopk(const Tensor scores, const Tensor bias,
              const int64_t num_expert_group, const int64_t topk_group,
              const int64_t topk, const bool renormalize,
              const double routed_scaling_factor, const int64_t scoring_func,
              Tensor topk_values, Tensor topk_indices)
      : num_tokens_{scores.ndim() == 2 ? scores.size(0) : 0},
        num_experts_{scores.ndim() == 2 ? scores.size(1) : 0},
        num_expert_group_{num_expert_group},
        topk_group_{topk_group},
        topk_{topk},
        renormalize_{renormalize},
        routed_scaling_factor_{routed_scaling_factor},
        scoring_func_{scoring_func},
        scores_dtype_{scores.dtype()},
        bias_dtype_{bias.dtype()},
        device_index_{scores.device().index()},
        scores_metadata_{scores},
        bias_metadata_{bias},
        topk_values_metadata_{topk_values},
        topk_indices_metadata_{topk_indices} {
    Validate(scores, bias, topk_values, topk_indices);
  }

  virtual void operator()(const Tensor scores, const Tensor bias,
                          const int64_t num_expert_group,
                          const int64_t topk_group, const int64_t topk,
                          const bool renormalize,
                          const double routed_scaling_factor,
                          const int64_t scoring_func, Tensor topk_values,
                          Tensor topk_indices) const = 0;

 protected:
  void ValidateCallMetadata(const Tensor scores, const Tensor bias,
                            const int64_t num_expert_group,
                            const int64_t topk_group, const int64_t topk,
                            const bool renormalize,
                            const double routed_scaling_factor,
                            const int64_t scoring_func, Tensor topk_values,
                            Tensor topk_indices) const {
    assert(num_expert_group == num_expert_group_ && topk_group == topk_group_ &&
           topk == topk_ && renormalize == renormalize_ &&
           routed_scaling_factor == routed_scaling_factor_ &&
           scoring_func == scoring_func_ &&
           "`GroupedTopk` attributes changed after descriptor creation");

    const std::equal_to<Tensor> same_metadata;
    const auto matches = same_metadata(scores_metadata_, scores) &&
                         same_metadata(bias_metadata_, bias) &&
                         same_metadata(topk_values_metadata_, topk_values) &&
                         same_metadata(topk_indices_metadata_, topk_indices);
    assert(matches && "`GroupedTopk` call metadata must match descriptor");
  }

  Tensor::Size num_tokens_{0};

  Tensor::Size num_experts_{0};

  int64_t num_expert_group_{0};

  int64_t topk_group_{0};

  int64_t topk_{0};

  bool renormalize_{false};

  double routed_scaling_factor_{1.0};

  int64_t scoring_func_{0};

  DataType scores_dtype_;

  DataType bias_dtype_;

  int device_index_{0};

 private:
  void Validate(const Tensor scores, const Tensor bias, Tensor topk_values,
                Tensor topk_indices) const {
    assert(scores.ndim() == 2 && "`GroupedTopk` requires 2D `scores`");
    assert((scores_dtype_ == DataType::kFloat16 ||
            scores_dtype_ == DataType::kBFloat16 ||
            scores_dtype_ == DataType::kFloat32) &&
           "`GroupedTopk` supports float16, bfloat16, and float32 `scores`");
    assert(scores.IsContiguous() &&
           "`GroupedTopk` requires contiguous `scores`");
    assert(num_tokens_ <=
               static_cast<Tensor::Size>(std::numeric_limits<int32_t>::max()) &&
           num_experts_ <=
               static_cast<Tensor::Size>(std::numeric_limits<int32_t>::max()) &&
           "`GroupedTopk` dimensions must fit int32 indexing");

    assert(bias.ndim() == 1 && bias.numel() == num_experts_ &&
           "`GroupedTopk` requires `bias` shape `[num_experts]`");
    assert((bias_dtype_ == DataType::kFloat16 ||
            bias_dtype_ == DataType::kBFloat16 ||
            bias_dtype_ == DataType::kFloat32) &&
           "`GroupedTopk` supports float16, bfloat16, and float32 `bias`");
    assert(bias.IsContiguous() && "`GroupedTopk` requires contiguous `bias`");

    assert(num_expert_group_ > 0 && num_expert_group_ <= 32 &&
           "`GroupedTopk` requires `num_expert_group` in `[1, 32]`");
    assert(topk_group_ > 0 && topk_group_ <= num_expert_group_ &&
           "`GroupedTopk` requires `topk_group` in `[1, num_expert_group]`");
    assert(num_experts_ > 0 &&
           num_experts_ % static_cast<Tensor::Size>(num_expert_group_) == 0 &&
           "`GroupedTopk` requires experts divisible by `num_expert_group`");
    assert(num_experts_ / static_cast<Tensor::Size>(num_expert_group_) >= 2 &&
           "`GroupedTopk` requires at least two experts per group");
    assert(topk_ > 0 && topk_ <= 32 &&
           topk_ <= topk_group_ * static_cast<int64_t>(num_experts_ /
                                                       num_expert_group_) &&
           "`GroupedTopk` requires `topk` in the selected-group capacity");
    assert((scoring_func_ == 0 || scoring_func_ == 1) &&
           "`GroupedTopk` requires `scoring_func` 0 (none) or 1 (sigmoid)");

    const Tensor::Shape output_shape{num_tokens_,
                                     static_cast<Tensor::Size>(topk_)};
    assert(topk_values.shape() == output_shape &&
           topk_indices.shape() == output_shape &&
           "`GroupedTopk` outputs must have shape `[num_tokens, topk]`");
    assert(topk_values.dtype() == DataType::kFloat32 &&
           "`GroupedTopk` requires float32 `topk_values`");
    assert(topk_indices.dtype() == DataType::kInt32 &&
           "`GroupedTopk` requires int32 `topk_indices`");
    assert(topk_values.IsContiguous() && topk_indices.IsContiguous() &&
           "`GroupedTopk` requires contiguous outputs");

    const auto same_device = [&](const Tensor tensor) {
      return tensor.device().type() == scores.device().type() &&
             tensor.device().index() == scores.device().index();
    };
    assert(same_device(bias) && same_device(topk_values) &&
           same_device(topk_indices) &&
           "`GroupedTopk` requires all tensors on the same device");
  }

  Tensor scores_metadata_;

  Tensor bias_metadata_;

  Tensor topk_values_metadata_;

  Tensor topk_indices_metadata_;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_GROUPED_TOPK_H_
