#ifndef INFINI_OPS_BASE_MOE_SUM_H_
#define INFINI_OPS_BASE_MOE_SUM_H_

#include <cassert>
#include <cstdint>
#include <limits>
#include <optional>

#include "operator.h"

namespace infini::ops {

// Aligned with vLLM `_moe_C::moe_sum`.
class MoeSum : public Operator<MoeSum> {
 public:
  MoeSum(const Tensor input, Tensor output)
      : MoeSum{input, std::nullopt, std::nullopt, output} {}

  MoeSum(const Tensor input, std::optional<Tensor> topk_ids,
         std::optional<Tensor> expert_map, Tensor output)
      : num_tokens_{input.ndim() == 3 ? input.size(0) : 0},
        topk_{input.ndim() == 3 ? input.size(1) : 0},
        hidden_size_{input.ndim() == 3 ? input.size(2) : 0},
        input_strides_{input.strides()},
        output_strides_{output.strides()},
        dtype_{input.dtype()},
        device_type_{input.device().type()},
        has_topk_ids_{topk_ids.has_value()},
        topk_ids_dtype_{topk_ids ? topk_ids->dtype() : DataType::kInt32},
        topk_ids_token_stride_{
            topk_ids && topk_ids->ndim() == 2 ? topk_ids->stride(0) : 0},
        topk_ids_slot_stride_{
            topk_ids && topk_ids->ndim() == 2 ? topk_ids->stride(1) : 0},
        has_expert_map_{expert_map.has_value()},
        expert_map_size_{expert_map ? expert_map->numel() : 0},
        expert_map_stride_{
            expert_map && expert_map->ndim() == 1 ? expert_map->stride(0) : 0},
        device_index_{input.device().index()} {
    assert(input.ndim() == 3 && output.ndim() == 2 &&
           "`MoeSum` requires `[num_tokens, topk, hidden_size]` input and "
           "`[num_tokens, hidden_size]` output");
    assert(output.size(0) == num_tokens_ && output.size(1) == hidden_size_ &&
           "`MoeSum` output shape is incompatible with the input");
    assert(topk_ > 0 && "`MoeSum` requires at least one top-k slot");
    assert((dtype_ == DataType::kFloat32 || dtype_ == DataType::kFloat16 ||
            dtype_ == DataType::kBFloat16) &&
           "`MoeSum` supports float32, float16, and bfloat16 inputs");
    assert(output.dtype() == dtype_ &&
           "`MoeSum` input and output dtypes must match");
    assert(output.IsContiguous() && "`MoeSum` requires contiguous output");

    constexpr auto kMaxSignedIndex =
        static_cast<Tensor::Size>(std::numeric_limits<int64_t>::max());
    assert(num_tokens_ <= kMaxSignedIndex && topk_ <= kMaxSignedIndex &&
           hidden_size_ <= kMaxSignedIndex &&
           "`MoeSum` dimensions must fit signed index arithmetic");
    assert(num_tokens_ <= std::numeric_limits<int>::max() &&
           "`MoeSum` token count exceeds the CUDA grid limit");
    assert(
        (hidden_size_ == 0 || num_tokens_ <= kMaxSignedIndex / hidden_size_) &&
        "`MoeSum` output size must fit signed index arithmetic");

    const auto offsets_fit = [](const auto& shape, const auto& strides) {
      uint64_t max_offset = 0;
      constexpr auto kMaxOffset =
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max());

      for (Tensor::Size dim = 0; dim < shape.size(); ++dim) {
        if (static_cast<uint64_t>(shape[dim]) > kMaxOffset) {
          return false;
        }
        if (strides[dim] < 0) {
          return false;
        }
        if (shape[dim] == 0) {
          continue;
        }

        const auto extent = static_cast<uint64_t>(shape[dim] - 1);
        const auto stride = static_cast<uint64_t>(strides[dim]);
        if (extent != 0 && stride > kMaxOffset / extent) {
          return false;
        }

        const auto term = extent * stride;
        if (term > kMaxOffset - max_offset) {
          return false;
        }
        max_offset += term;
      }

      return true;
    };
    assert(offsets_fit(input.shape(), input.strides()) &&
           "`MoeSum` input requires non-negative strides with signed offsets");

    const auto same_device_as_input = [&](const Tensor tensor) {
      return tensor.device().type() == input.device().type() &&
             tensor.device().index() == input.device().index();
    };
    assert(same_device_as_input(output) &&
           "`MoeSum` input and output must be on the same device");
    assert((!expert_map || topk_ids) &&
           "`MoeSum` expert_map requires topk_ids");

    if (topk_ids) {
      assert(topk_ids->ndim() == 2 && topk_ids->size(0) == num_tokens_ &&
             topk_ids->size(1) == topk_ &&
             "`MoeSum` topk_ids must have shape `[num_tokens, topk]`");
      assert((topk_ids_dtype_ == DataType::kInt32 ||
              topk_ids_dtype_ == DataType::kInt64) &&
             "`MoeSum` topk_ids must have int32 or int64 dtype");
      assert(same_device_as_input(*topk_ids) &&
             "`MoeSum` topk_ids must be on the input device");
      assert(offsets_fit(topk_ids->shape(), topk_ids->strides()) &&
             "`MoeSum` topk_ids requires non-negative strides with signed "
             "offsets");
    }

    if (expert_map) {
      assert(expert_map->ndim() == 1 &&
             expert_map->dtype() == DataType::kInt32 &&
             "`MoeSum` expert_map must be a 1D int32 tensor");
      assert(same_device_as_input(*expert_map) &&
             "`MoeSum` expert_map must be on the input device");
      assert(offsets_fit(expert_map->shape(), expert_map->strides()) &&
             "`MoeSum` expert_map requires non-negative strides with signed "
             "offsets");
    }
  }

  void operator()(const Tensor input, Tensor output) const {
    (*this)(input, std::nullopt, std::nullopt, output);
  }

  virtual void operator()(const Tensor input, std::optional<Tensor> topk_ids,
                          std::optional<Tensor> expert_map,
                          Tensor output) const = 0;

 protected:
  void ValidateCallMetadata(const Tensor input, std::optional<Tensor> topk_ids,
                            std::optional<Tensor> expert_map,
                            const Tensor output) const {
    const auto same_device_as_descriptor = [&](const Tensor tensor) {
      return tensor.device().type() == device_type_ &&
             tensor.device().index() == device_index_;
    };
    auto matches =
        input.ndim() == 3 && input.size(0) == num_tokens_ &&
        input.size(1) == topk_ && input.size(2) == hidden_size_ &&
        input.strides() == input_strides_ && input.dtype() == dtype_ &&
        same_device_as_descriptor(input) && output.ndim() == 2 &&
        output.size(0) == num_tokens_ && output.size(1) == hidden_size_ &&
        output.strides() == output_strides_ && output.dtype() == dtype_ &&
        same_device_as_descriptor(output) &&
        topk_ids.has_value() == has_topk_ids_ &&
        expert_map.has_value() == has_expert_map_;

    if (matches && topk_ids) {
      matches = topk_ids->ndim() == 2 && topk_ids->size(0) == num_tokens_ &&
                topk_ids->size(1) == topk_ &&
                topk_ids->stride(0) == topk_ids_token_stride_ &&
                topk_ids->stride(1) == topk_ids_slot_stride_ &&
                topk_ids->dtype() == topk_ids_dtype_ &&
                same_device_as_descriptor(*topk_ids);
    }

    if (matches && expert_map) {
      matches = expert_map->ndim() == 1 &&
                expert_map->numel() == expert_map_size_ &&
                expert_map->stride(0) == expert_map_stride_ &&
                expert_map->dtype() == DataType::kInt32 &&
                same_device_as_descriptor(*expert_map);
    }

    assert(matches && "`MoeSum` call metadata must match descriptor");
  }

  Tensor::Size num_tokens_{0};

  Tensor::Size topk_{0};

  Tensor::Size hidden_size_{0};

  Tensor::Strides input_strides_;

  Tensor::Strides output_strides_;

  DataType dtype_;

  Device::Type device_type_;

  bool has_topk_ids_{false};

  DataType topk_ids_dtype_;

  Tensor::Stride topk_ids_token_stride_{0};

  Tensor::Stride topk_ids_slot_stride_{0};

  bool has_expert_map_{false};

  Tensor::Size expert_map_size_{0};

  Tensor::Stride expert_map_stride_{0};

  int device_index_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_MOE_SUM_H_
