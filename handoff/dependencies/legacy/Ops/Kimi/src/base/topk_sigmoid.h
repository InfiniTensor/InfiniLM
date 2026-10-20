#ifndef INFINI_OPS_BASE_TOPK_SIGMOID_H_
#define INFINI_OPS_BASE_TOPK_SIGMOID_H_

#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>

#include "operator.h"

namespace infini::ops {

// Aligned with vLLM `_moe_C::topk_sigmoid`.
class TopkSigmoid : public Operator<TopkSigmoid> {
 public:
  TopkSigmoid(const Tensor gating_output,
              std::optional<Tensor> e_score_correction_bias,
              std::optional<Tensor> is_padding, const bool renormalize,
              const double routed_scaling_factor, Tensor topk_weights,
              Tensor topk_ids, Tensor token_expert_indices)
      : num_tokens_{gating_output.ndim() == 2 ? gating_output.size(0) : 0},
        num_experts_{gating_output.ndim() == 2 ? gating_output.size(1) : 0},
        topk_{topk_weights.ndim() == 2 ? topk_weights.size(1) : 0},
        input_dtype_{gating_output.dtype()},
        index_dtype_{topk_ids.dtype()},
        renormalize_{renormalize},
        routed_scaling_factor_{routed_scaling_factor},
        device_index_{gating_output.device().index()},
        gating_output_metadata_{gating_output},
        e_score_correction_bias_metadata_{e_score_correction_bias},
        is_padding_metadata_{is_padding},
        topk_weights_metadata_{topk_weights},
        topk_ids_metadata_{topk_ids},
        token_expert_indices_metadata_{token_expert_indices} {
    Validate(gating_output, e_score_correction_bias, is_padding, topk_weights,
             topk_ids, token_expert_indices);
  }

  virtual void operator()(const Tensor gating_output,
                          std::optional<Tensor> e_score_correction_bias,
                          std::optional<Tensor> is_padding,
                          const bool renormalize,
                          const double routed_scaling_factor,
                          Tensor topk_weights, Tensor topk_ids,
                          Tensor token_expert_indices) const = 0;

 protected:
  void ValidateCallMetadata(const Tensor gating_output,
                            std::optional<Tensor> e_score_correction_bias,
                            std::optional<Tensor> is_padding,
                            const bool renormalize,
                            const double routed_scaling_factor,
                            Tensor topk_weights, Tensor topk_ids,
                            Tensor token_expert_indices) const {
    assert(renormalize == renormalize_ &&
           routed_scaling_factor == routed_scaling_factor_ &&
           "`TopkSigmoid` attributes changed after descriptor creation");

    const std::equal_to<Tensor> same_metadata;
    const auto optional_matches = [&](const std::optional<Tensor>& expected,
                                      const std::optional<Tensor>& actual) {
      return expected.has_value() == actual.has_value() &&
             (!expected || same_metadata(*expected, *actual));
    };
    const auto matches =
        same_metadata(gating_output_metadata_, gating_output) &&
        optional_matches(e_score_correction_bias_metadata_,
                         e_score_correction_bias) &&
        optional_matches(is_padding_metadata_, is_padding) &&
        same_metadata(topk_weights_metadata_, topk_weights) &&
        same_metadata(topk_ids_metadata_, topk_ids) &&
        same_metadata(token_expert_indices_metadata_, token_expert_indices);
    assert(matches && "`TopkSigmoid` call metadata must match descriptor");
  }

  Tensor::Size num_tokens_{0};

  Tensor::Size num_experts_{0};

  Tensor::Size topk_{0};

  DataType input_dtype_;

  DataType index_dtype_;

  bool renormalize_{false};

  double routed_scaling_factor_{1.0};

  int device_index_{0};

 private:
  void Validate(const Tensor gating_output,
                std::optional<Tensor> e_score_correction_bias,
                std::optional<Tensor> is_padding, Tensor topk_weights,
                Tensor topk_ids, Tensor token_expert_indices) const {
    assert(gating_output.ndim() == 2 &&
           "`TopkSigmoid` requires 2D `gating_output`");
    assert((input_dtype_ == DataType::kFloat32 ||
            input_dtype_ == DataType::kFloat16 ||
            input_dtype_ == DataType::kBFloat16) &&
           "`TopkSigmoid` supports float32, float16, and bfloat16 input");
    assert(gating_output.IsContiguous() &&
           "`TopkSigmoid` requires contiguous `gating_output`");
    assert(num_experts_ > 0 && topk_ > 0 && topk_ <= num_experts_ &&
           "`TopkSigmoid` requires `topk` in `[1, num_experts]`");
    assert(std::isfinite(routed_scaling_factor_) &&
           "`TopkSigmoid` requires a finite `routed_scaling_factor`");
    assert(num_tokens_ <=
               static_cast<Tensor::Size>(std::numeric_limits<int32_t>::max()) &&
           num_experts_ <=
               static_cast<Tensor::Size>(std::numeric_limits<int32_t>::max()) &&
           topk_ <=
               static_cast<Tensor::Size>(std::numeric_limits<int32_t>::max()) &&
           "`TopkSigmoid` dimensions must fit int32 indexing");
    assert(num_tokens_ <=
               static_cast<Tensor::Size>(std::numeric_limits<int32_t>::max()) /
                   topk_ &&
           "`TopkSigmoid` output indices must fit int32 indexing");

    const Tensor::Shape output_shape{num_tokens_, topk_};
    assert(topk_weights.shape() == output_shape &&
           topk_ids.shape() == output_shape &&
           token_expert_indices.shape() == output_shape &&
           "`TopkSigmoid` outputs must have shape `[num_tokens, topk]`");
    assert(topk_weights.dtype() == DataType::kFloat32 &&
           "`TopkSigmoid` requires float32 `topk_weights`");
    assert((index_dtype_ == DataType::kInt32 ||
            index_dtype_ == DataType::kUInt32 ||
            index_dtype_ == DataType::kInt64) &&
           "`TopkSigmoid` requires int32, uint32, or int64 `topk_ids`");
    assert(token_expert_indices.dtype() == DataType::kInt32 &&
           "`TopkSigmoid` requires int32 `token_expert_indices`");
    assert(topk_weights.IsContiguous() && topk_ids.IsContiguous() &&
           token_expert_indices.IsContiguous() &&
           "`TopkSigmoid` requires contiguous outputs");

    const auto same_device = [&](const Tensor tensor) {
      return tensor.device().type() == gating_output.device().type() &&
             tensor.device().index() == gating_output.device().index();
    };
    assert(same_device(topk_weights) && same_device(topk_ids) &&
           same_device(token_expert_indices) &&
           "`TopkSigmoid` requires all tensors on the same device");

    if (e_score_correction_bias) {
      assert(e_score_correction_bias->ndim() == 1 &&
             e_score_correction_bias->numel() == num_experts_ &&
             "`TopkSigmoid` requires `e_score_correction_bias` shape "
             "`[num_experts]`");
      assert(e_score_correction_bias->dtype() == DataType::kFloat32 &&
             e_score_correction_bias->IsContiguous() &&
             "`TopkSigmoid` requires contiguous float32 "
             "`e_score_correction_bias`");
      assert(same_device(*e_score_correction_bias) &&
             "`TopkSigmoid` requires `e_score_correction_bias` on the input "
             "device");
    }

    if (is_padding) {
      assert(is_padding->ndim() == 1 && is_padding->numel() == num_tokens_ &&
             "`TopkSigmoid` requires `is_padding` shape `[num_tokens]`");
      assert(is_padding->dtype() == DataType::kBool &&
             is_padding->IsContiguous() &&
             "`TopkSigmoid` requires contiguous bool `is_padding`");
      assert(same_device(*is_padding) &&
             "`TopkSigmoid` requires `is_padding` on the input device");
    }
  }

  Tensor gating_output_metadata_;

  std::optional<Tensor> e_score_correction_bias_metadata_;

  std::optional<Tensor> is_padding_metadata_;

  Tensor topk_weights_metadata_;

  Tensor topk_ids_metadata_;

  Tensor token_expert_indices_metadata_;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_TOPK_SIGMOID_H_
