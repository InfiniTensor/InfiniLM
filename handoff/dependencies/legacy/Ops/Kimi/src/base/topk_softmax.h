#ifndef INFINI_OPS_BASE_TOPK_SOFTMAX_H_
#define INFINI_OPS_BASE_TOPK_SOFTMAX_H_

#include <cassert>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <cmath>
#include <stdexcept>

#include "operator.h"

namespace infini::ops {

class TopkSoftmaxConfig : public Config {
 public:
  explicit TopkSoftmaxConfig(float softcap = 0.0f, float min_sum = 0.0f)
      : softcap_{softcap}, min_sum_{min_sum} {
    if (!std::isfinite(softcap) || softcap < 0.0f) {
      throw std::invalid_argument("TopkSoftmax softcap must be finite and nonnegative");
    }
    if (!std::isfinite(min_sum) || min_sum < 0.0f) {
      throw std::invalid_argument("TopkSoftmax min_sum must be finite and nonnegative");
    }
  }

  std::unique_ptr<Config> Clone() const override {
    return std::make_unique<TopkSoftmaxConfig>(*this);
  }

  bool requires_config_aware_provider() const override {
    return softcap_ != 0.0f || min_sum_ != 0.0f;
  }
  float softcap() const { return softcap_; }
  float min_sum() const { return min_sum_; }

  static float SoftcapFrom(const Config& config) {
    const auto* typed = dynamic_cast<const TopkSoftmaxConfig*>(&config);
    return typed ? typed->softcap() : 0.0f;
  }

  static float MinSumFrom(const Config& config) {
    const auto* typed = dynamic_cast<const TopkSoftmaxConfig*>(&config);
    return typed ? typed->min_sum() : 0.0f;
  }

 private:
  float softcap_;
  float min_sum_;
};

// Aligned with vLLM `_moe_C::topk_softmax`.
class TopkSoftmax : public Operator<TopkSoftmax> {
 public:
  TopkSoftmax(const Tensor gating_output, std::optional<Tensor> bias,
              std::optional<Tensor> is_padding, const bool renormalize,
              Tensor topk_weights, Tensor topk_indices,
              Tensor token_expert_indices)
      : num_tokens_{gating_output.ndim() == 2 ? gating_output.size(0) : 0},
        num_experts_{gating_output.ndim() == 2 ? gating_output.size(1) : 0},
        topk_{topk_weights.ndim() == 2 ? topk_weights.size(1) : 0},
        input_dtype_{gating_output.dtype()},
        index_dtype_{topk_indices.dtype()},
        renormalize_{renormalize},
        device_index_{gating_output.device().index()},
        gating_output_metadata_{gating_output},
        bias_metadata_{bias},
        is_padding_metadata_{is_padding},
        topk_weights_metadata_{topk_weights},
        topk_indices_metadata_{topk_indices},
        token_expert_indices_metadata_{token_expert_indices} {
    Validate(gating_output, bias, is_padding, topk_weights, topk_indices,
             token_expert_indices);
  }

  virtual void operator()(const Tensor gating_output,
                          std::optional<Tensor> bias,
                          std::optional<Tensor> is_padding,
                          const bool renormalize, Tensor topk_weights,
                          Tensor topk_indices,
                          Tensor token_expert_indices) const = 0;

 protected:
  float softcap_{0.0f};

  float min_sum_{0.0f};

  void ValidateCallMetadata(const Tensor gating_output,
                            std::optional<Tensor> bias,
                            std::optional<Tensor> is_padding,
                            const bool renormalize, Tensor topk_weights,
                            Tensor topk_indices,
                            Tensor token_expert_indices) const {
    assert(renormalize == renormalize_ &&
           "`TopkSoftmax` attributes changed after descriptor creation");

    const std::equal_to<Tensor> same_metadata;
    const auto optional_matches = [&](const std::optional<Tensor>& expected,
                                      const std::optional<Tensor>& actual) {
      return expected.has_value() == actual.has_value() &&
             (!expected || same_metadata(*expected, *actual));
    };
    const auto matches =
        same_metadata(gating_output_metadata_, gating_output) &&
        optional_matches(bias_metadata_, bias) &&
        optional_matches(is_padding_metadata_, is_padding) &&
        same_metadata(topk_weights_metadata_, topk_weights) &&
        same_metadata(topk_indices_metadata_, topk_indices) &&
        same_metadata(token_expert_indices_metadata_, token_expert_indices);
    assert(matches && "`TopkSoftmax` call metadata must match descriptor");
  }

  Tensor::Size num_tokens_{0};

  Tensor::Size num_experts_{0};

  Tensor::Size topk_{0};

  DataType input_dtype_;

  DataType index_dtype_;

  bool renormalize_{false};

  int device_index_{0};

 private:
  void Validate(const Tensor gating_output, std::optional<Tensor> bias,
                std::optional<Tensor> is_padding, Tensor topk_weights,
                Tensor topk_indices, Tensor token_expert_indices) const {
    assert(gating_output.ndim() == 2 &&
           "`TopkSoftmax` requires 2D `gating_output`");
    assert((input_dtype_ == DataType::kFloat32 ||
            input_dtype_ == DataType::kFloat16 ||
            input_dtype_ == DataType::kBFloat16) &&
           "`TopkSoftmax` supports float32, float16, and bfloat16 input");
    assert(gating_output.IsContiguous() &&
           "`TopkSoftmax` requires contiguous `gating_output`");
    assert(num_experts_ > 0 && topk_ > 0 && topk_ <= num_experts_ &&
           "`TopkSoftmax` requires `topk` in `[1, num_experts]`");
    assert(num_tokens_ <=
               static_cast<Tensor::Size>(std::numeric_limits<int32_t>::max()) &&
           num_experts_ <=
               static_cast<Tensor::Size>(std::numeric_limits<int32_t>::max()) &&
           topk_ <=
               static_cast<Tensor::Size>(std::numeric_limits<int32_t>::max()) &&
           "`TopkSoftmax` dimensions must fit int32 indexing");
    assert(num_tokens_ <=
               static_cast<Tensor::Size>(std::numeric_limits<int32_t>::max()) /
                   topk_ &&
           "`TopkSoftmax` output indices must fit int32 indexing");

    const Tensor::Shape output_shape{num_tokens_, topk_};
    assert(topk_weights.shape() == output_shape &&
           topk_indices.shape() == output_shape &&
           token_expert_indices.shape() == output_shape &&
           "`TopkSoftmax` outputs must have shape `[num_tokens, topk]`");
    assert(topk_weights.dtype() == DataType::kFloat32 &&
           "`TopkSoftmax` requires float32 `topk_weights`");
    assert((index_dtype_ == DataType::kInt32 ||
            index_dtype_ == DataType::kUInt32 ||
            index_dtype_ == DataType::kInt64) &&
           "`TopkSoftmax` requires int32, uint32, or int64 `topk_indices`");
    assert(token_expert_indices.dtype() == DataType::kInt32 &&
           "`TopkSoftmax` requires int32 `token_expert_indices`");
    assert(topk_weights.IsContiguous() && topk_indices.IsContiguous() &&
           token_expert_indices.IsContiguous() &&
           "`TopkSoftmax` requires contiguous outputs");

    const auto same_device = [&](const Tensor tensor) {
      return tensor.device().type() == gating_output.device().type() &&
             tensor.device().index() == gating_output.device().index();
    };
    assert(same_device(topk_weights) && same_device(topk_indices) &&
           same_device(token_expert_indices) &&
           "`TopkSoftmax` requires all tensors on the same device");

    if (bias) {
      assert(bias->ndim() == 1 && bias->numel() == num_experts_ &&
             "`TopkSoftmax` requires `bias` shape `[num_experts]`");
      assert(bias->dtype() == DataType::kFloat32 && bias->IsContiguous() &&
             "`TopkSoftmax` requires contiguous float32 `bias`");
      assert(same_device(*bias) &&
             "`TopkSoftmax` requires `bias` on the input device");
    }

    if (is_padding) {
      assert(is_padding->ndim() == 1 && is_padding->numel() == num_tokens_ &&
             "`TopkSoftmax` requires `is_padding` shape `[num_tokens]`");
      assert(is_padding->dtype() == DataType::kBool &&
             is_padding->IsContiguous() &&
             "`TopkSoftmax` requires contiguous bool `is_padding`");
      assert(same_device(*is_padding) &&
             "`TopkSoftmax` requires `is_padding` on the input device");
    }
  }

  Tensor gating_output_metadata_;

  std::optional<Tensor> bias_metadata_;

  std::optional<Tensor> is_padding_metadata_;

  Tensor topk_weights_metadata_;

  Tensor topk_indices_metadata_;

  Tensor token_expert_indices_metadata_;
};

template <>
struct CacheKeyBuilder<TopkSoftmax> {
  template <typename... Args>
  detail::CacheKey operator()(const Config& config, const Args&... args) const {
    return detail::CacheKey::Build(config.implementation_index(),
                                   TopkSoftmaxConfig::SoftcapFrom(config),
                                   TopkSoftmaxConfig::MinSumFrom(config), args...);
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_TOPK_SOFTMAX_H_
