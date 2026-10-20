#ifndef INFINI_OPS_BASE_MOE_ALIGN_BLOCK_SIZE_H_
#define INFINI_OPS_BASE_MOE_ALIGN_BLOCK_SIZE_H_

#include <cassert>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>

#include "operator.h"

namespace infini::ops {

// Align routed token indices into expert-specific blocks following vLLM's
// low-level `moe_align_block_size` operator.
class MoeAlignBlockSize : public Operator<MoeAlignBlockSize> {
 public:
  MoeAlignBlockSize(const Tensor topk_ids,
                    const std::optional<Tensor> expert_map,
                    const int64_t num_experts, const int64_t block_size,
                    Tensor sorted_token_ids, Tensor experts_ids,
                    Tensor num_tokens_post_pad)
      : topk_ids_metadata_{topk_ids},
        expert_map_metadata_{expert_map},
        sorted_token_ids_metadata_{sorted_token_ids},
        experts_ids_metadata_{experts_ids},
        num_tokens_post_pad_metadata_{num_tokens_post_pad},
        numel_{topk_ids.numel()},
        num_experts_{num_experts},
        block_size_{block_size},
        sorted_token_ids_size_{sorted_token_ids.numel()},
        experts_ids_size_{experts_ids.numel()} {
    Validate(topk_ids, expert_map, sorted_token_ids, experts_ids,
             num_tokens_post_pad);
  }

  void operator()(const Tensor topk_ids, const std::optional<Tensor> expert_map,
                  const int64_t num_experts, const int64_t block_size,
                  Tensor sorted_token_ids, Tensor experts_ids,
                  Tensor num_tokens_post_pad) const {
    ValidateInvocation(topk_ids, expert_map, num_experts, block_size,
                       sorted_token_ids, experts_ids, num_tokens_post_pad);
    Run(topk_ids, expert_map, num_experts, block_size, sorted_token_ids,
        experts_ids, num_tokens_post_pad);
  }

 protected:
  void ValidateInvocation(const Tensor topk_ids,
                          const std::optional<Tensor> maybe_expert_map,
                          const int64_t num_experts, const int64_t block_size,
                          Tensor sorted_token_ids, Tensor experts_ids,
                          Tensor num_tokens_post_pad) const {
    assert(num_experts == num_experts_ && block_size == block_size_ &&
           "`MoeAlignBlockSize` attributes changed after descriptor creation");

    assert(CallMetadataMatches(topk_ids, maybe_expert_map, sorted_token_ids,
                               experts_ids, num_tokens_post_pad) &&
           "`MoeAlignBlockSize` tensor metadata differs from its descriptor");
  }

  bool CallMetadataMatches(const Tensor topk_ids,
                           const std::optional<Tensor> maybe_expert_map,
                           const Tensor sorted_token_ids,
                           const Tensor experts_ids,
                           const Tensor num_tokens_post_pad) const {
    const std::equal_to<Tensor> same_metadata;
    const auto same_expert_map_metadata =
        expert_map_metadata_.has_value() == maybe_expert_map.has_value() &&
        (!expert_map_metadata_ ||
         same_metadata(*expert_map_metadata_, *maybe_expert_map));

    return same_metadata(topk_ids_metadata_, topk_ids) &&
           same_expert_map_metadata &&
           same_metadata(sorted_token_ids_metadata_, sorted_token_ids) &&
           same_metadata(experts_ids_metadata_, experts_ids) &&
           same_metadata(num_tokens_post_pad_metadata_, num_tokens_post_pad);
  }

  void Validate(const Tensor topk_ids,
                const std::optional<Tensor> maybe_expert_map,
                Tensor sorted_token_ids, Tensor experts_ids,
                Tensor num_tokens_post_pad) const {
    assert(topk_ids.ndim() == 2 &&
           "`MoeAlignBlockSize` requires 2D `topk_ids`");
    assert(topk_ids.dtype() == DataType::kInt32 &&
           "`MoeAlignBlockSize` currently requires int32 `topk_ids`");
    assert(topk_ids.IsContiguous() &&
           "`MoeAlignBlockSize` requires contiguous `topk_ids`");
    assert(num_experts_ > 0 && num_experts_ < 1024 &&
           "`MoeAlignBlockSize` requires `num_experts` in [1, 1023]");
    assert(block_size_ > 0 &&
           "`MoeAlignBlockSize` requires a positive `block_size`");
    assert(block_size_ <= std::numeric_limits<int32_t>::max() &&
           numel_ <=
               static_cast<Tensor::Size>(std::numeric_limits<int32_t>::max()) &&
           "`MoeAlignBlockSize` requires int32-addressable token indices");

    const auto same_device_as_topk_ids = [&](const Tensor tensor) {
      return tensor.device().type() == topk_ids.device().type() &&
             tensor.device().index() == topk_ids.device().index();
    };
    assert(same_device_as_topk_ids(sorted_token_ids) &&
           same_device_as_topk_ids(experts_ids) &&
           same_device_as_topk_ids(num_tokens_post_pad) &&
           "`MoeAlignBlockSize` requires all tensors on the same device");

    assert(sorted_token_ids.ndim() == 1 && experts_ids.ndim() == 1 &&
           num_tokens_post_pad.ndim() == 1 &&
           num_tokens_post_pad.numel() == 1 &&
           "`MoeAlignBlockSize` requires 1D output tensors");
    assert(sorted_token_ids.dtype() == DataType::kInt32 &&
           experts_ids.dtype() == DataType::kInt32 &&
           num_tokens_post_pad.dtype() == DataType::kInt32 &&
           "`MoeAlignBlockSize` requires int32 output tensors");
    assert(sorted_token_ids.IsContiguous() && experts_ids.IsContiguous() &&
           num_tokens_post_pad.IsContiguous() &&
           "`MoeAlignBlockSize` requires contiguous output tensors");

    const auto num_experts = static_cast<Tensor::Size>(num_experts_);
    const auto block_size = static_cast<Tensor::Size>(block_size_);
    auto required_sorted_size = numel_ + num_experts * (block_size - 1);
    if (numel_ < num_experts) {
      const auto small_input_size = numel_ * block_size;
      required_sorted_size = small_input_size < required_sorted_size
                                 ? small_input_size
                                 : required_sorted_size;
    }
    const auto required_experts_size =
        (required_sorted_size + block_size - 1) / block_size;
    assert(required_sorted_size <=
               static_cast<Tensor::Size>(std::numeric_limits<int32_t>::max()) &&
           "`MoeAlignBlockSize` requires int32-addressable padded indices");
    assert(sorted_token_ids_size_ >= required_sorted_size &&
           experts_ids_size_ >= required_experts_size &&
           "`MoeAlignBlockSize` output tensors are too small");

    if (maybe_expert_map) {
      assert(
          maybe_expert_map->ndim() == 1 && maybe_expert_map->numel() > 0 &&
          maybe_expert_map->numel() <=
              static_cast<Tensor::Size>(std::numeric_limits<int32_t>::max()) &&
          "`MoeAlignBlockSize` requires an int32-addressable `expert_map`");
      assert(maybe_expert_map->dtype() == DataType::kInt32 &&
             "`MoeAlignBlockSize` currently requires int32 `expert_map`");
      assert(maybe_expert_map->IsContiguous() &&
             "`MoeAlignBlockSize` requires contiguous `expert_map`");
      assert(same_device_as_topk_ids(*maybe_expert_map) &&
             "`MoeAlignBlockSize` requires `expert_map` on the input device");
    }
  }

  virtual void Run(const Tensor topk_ids,
                   const std::optional<Tensor> maybe_expert_map,
                   const int64_t num_experts, const int64_t block_size,
                   Tensor sorted_token_ids, Tensor experts_ids,
                   Tensor num_tokens_post_pad) const = 0;

  Tensor topk_ids_metadata_;

  std::optional<Tensor> expert_map_metadata_;

  Tensor sorted_token_ids_metadata_;

  Tensor experts_ids_metadata_;

  Tensor num_tokens_post_pad_metadata_;

  Tensor::Size numel_{0};

  int64_t num_experts_{0};

  int64_t block_size_{0};

  Tensor::Size sorted_token_ids_size_{0};

  Tensor::Size experts_ids_size_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_MOE_ALIGN_BLOCK_SIZE_H_
