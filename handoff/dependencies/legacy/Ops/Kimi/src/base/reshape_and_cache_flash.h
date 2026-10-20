#ifndef INFINI_OPS_BASE_RESHAPE_AND_CACHE_FLASH_H_
#define INFINI_OPS_BASE_RESHAPE_AND_CACHE_FLASH_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>

#include "data_type.h"
#include "operator.h"

namespace infini::ops {

// Aligned with vLLM `_custom_ops.reshape_and_cache_flash`.
class ReshapeAndCacheFlash : public Operator<ReshapeAndCacheFlash> {
 public:
  static bool supports(const Tensor& key, const Tensor& value,
                       const Tensor& slots, const Tensor& k_scale,
                       const Tensor& v_scale, const std::string& mode,
                       const Tensor& key_cache, const Tensor& value_cache) {
    if (key.ndim() != 3 || value.ndim() != 3 || slots.ndim() != 1 ||
        key_cache.ndim() != 4 || value_cache.ndim() != 4) return false;
    return key.shape() == value.shape() && key.dtype() == value.dtype() &&
           (key.dtype() == DataType::kFloat16 || key.dtype() == DataType::kBFloat16 ||
            key.dtype() == DataType::kFloat32) && slots.dtype() == DataType::kInt64 &&
           slots.stride(0) == 1 && slots.size(0) <= key.size(0) && key.size(1) > 0 &&
           key.size(2) > 0 && key_cache.shape() == value_cache.shape() &&
           key_cache.size(1) > 0 && key_cache.size(2) == key.size(1) &&
           key_cache.size(3) == key.size(2) && key.stride(2) == 1 &&
           value.stride(2) == 1 && key_cache.stride(3) == 1 && value_cache.stride(3) == 1 &&
           mode == "auto" && key_cache.dtype() == key.dtype() && value_cache.dtype() == key.dtype() &&
           !key_cache.HasBroadcastDim() && !value_cache.HasBroadcastDim() &&
           k_scale.shape() == v_scale.shape() &&
           (k_scale.numel() == 1 || k_scale.numel() == key.size(1)) &&
           k_scale.dtype() == DataType::kFloat32 && v_scale.dtype() == DataType::kFloat32;
  }

  ReshapeAndCacheFlash(const Tensor key, const Tensor value,
                       const Tensor slot_mapping, const Tensor k_scale,
                       const Tensor v_scale, const std::string kv_cache_dtype,
                       Tensor key_cache, Tensor value_cache)
      : dtype_{key.dtype()},
        num_tokens_{slot_mapping.size(0)},
        num_heads_{key.size(1)},
        head_size_{key.size(2)},
        block_size_{key_cache.size(1)},
        key_token_stride_{key.stride(0)},
        value_token_stride_{value.stride(0)},
        key_head_stride_{key.stride(1)},
        value_head_stride_{value.stride(1)},
        key_cache_block_stride_{key_cache.stride(0)},
        value_cache_block_stride_{value_cache.stride(0)},
        key_cache_page_stride_{key_cache.stride(1)},
        value_cache_page_stride_{value_cache.stride(1)},
        key_cache_head_stride_{key_cache.stride(2)},
        value_cache_head_stride_{value_cache.stride(2)} {
    assert(key.ndim() == 3 && value.ndim() == 3 &&
           "`ReshapeAndCacheFlash` requires 3D `key` and `value`");
    assert(key.shape() == value.shape() && key.dtype() == value.dtype() &&
           "`ReshapeAndCacheFlash` requires `key` and `value` to have matching "
           "shapes and dtypes");
    assert((dtype_ == DataType::kFloat16 || dtype_ == DataType::kBFloat16 ||
            dtype_ == DataType::kFloat32) &&
           "`ReshapeAndCacheFlash` supports float16, bfloat16, and float32");
    assert(slot_mapping.ndim() == 1 &&
           slot_mapping.dtype() == DataType::kInt64 &&
           slot_mapping.stride(0) == 1 &&
           "`ReshapeAndCacheFlash` requires contiguous int64 `slot_mapping`");
    assert(num_tokens_ <= key.size(0) &&
           "`ReshapeAndCacheFlash` requires enough `key` and `value` rows");
    assert(num_heads_ > 0 && head_size_ > 0 &&
           "`ReshapeAndCacheFlash` requires non-empty head dimensions");
    assert(key_cache.ndim() == 4 && value_cache.ndim() == 4 &&
           key_cache.shape() == value_cache.shape() &&
           "`ReshapeAndCacheFlash` requires matching 4D caches");
    assert(block_size_ > 0 && key_cache.size(2) == num_heads_ &&
           key_cache.size(3) == head_size_ &&
           "`ReshapeAndCacheFlash` cache shape must be "
           "[`num_blocks`, `block_size`, `num_heads`, `head_size`]");
    assert(key.stride(2) == 1 && value.stride(2) == 1 &&
           key_cache.stride(3) == 1 && value_cache.stride(3) == 1 &&
           "`ReshapeAndCacheFlash` requires contiguous head dimensions");
    assert(key.device() == value.device() &&
           key.device() == slot_mapping.device() &&
           key.device() == k_scale.device() &&
           key.device() == v_scale.device() &&
           key.device() == key_cache.device() &&
           key.device() == value_cache.device() &&
           "`ReshapeAndCacheFlash` tensors must be on the same device");
    assert(k_scale.shape() == v_scale.shape() &&
           (k_scale.numel() == 1 || k_scale.numel() == num_heads_) &&
           k_scale.dtype() == DataType::kFloat32 &&
           v_scale.dtype() == DataType::kFloat32 &&
           "`ReshapeAndCacheFlash` scales must be float32 scalar or per-head "
           "tensors");
    assert(kv_cache_dtype == "auto" && key_cache.dtype() == dtype_ &&
           value_cache.dtype() == dtype_ &&
           "`ReshapeAndCacheFlash` currently supports `auto` cache dtype");
    assert(!key_cache.HasBroadcastDim() && !value_cache.HasBroadcastDim() &&
           "`ReshapeAndCacheFlash` caches must not have broadcast dimensions");
  }

  virtual void operator()(const Tensor key, const Tensor value,
                          const Tensor slot_mapping, const Tensor k_scale,
                          const Tensor v_scale,
                          const std::string kv_cache_dtype, Tensor key_cache,
                          Tensor value_cache) const = 0;

 protected:
  DataType dtype_;

  std::size_t num_tokens_{0};

  std::size_t num_heads_{0};

  std::size_t head_size_{0};

  std::size_t block_size_{0};

  Tensor::Stride key_token_stride_{0};

  Tensor::Stride value_token_stride_{0};

  Tensor::Stride key_head_stride_{0};

  Tensor::Stride value_head_stride_{0};

  Tensor::Stride key_cache_block_stride_{0};

  Tensor::Stride value_cache_block_stride_{0};

  Tensor::Stride key_cache_page_stride_{0};

  Tensor::Stride value_cache_page_stride_{0};

  Tensor::Stride key_cache_head_stride_{0};

  Tensor::Stride value_cache_head_stride_{0};
};

}  // namespace infini::ops

#endif
