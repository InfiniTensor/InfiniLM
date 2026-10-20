#ifndef INFINI_OPS_BASE_RESHAPE_AND_CACHE_H_
#define INFINI_OPS_BASE_RESHAPE_AND_CACHE_H_

#include <string>

#include "data_type.h"
#include "operator.h"

namespace infini::ops {

class ReshapeAndCache : public Operator<ReshapeAndCache> {
 public:
  ReshapeAndCache(const Tensor key, const Tensor value, Tensor key_cache,
                  Tensor value_cache, const Tensor slot_mapping,
                  const Tensor k_scale, const Tensor v_scale,
                  const std::string kv_cache_dtype)
      : key_shape_{key.shape()},
        value_shape_{value.shape()},
        key_cache_shape_{key_cache.shape()},
        value_cache_shape_{value_cache.shape()},
        slot_mapping_shape_{slot_mapping.shape()},
        k_scale_shape_{k_scale.shape()},
        v_scale_shape_{v_scale.shape()},
        key_strides_{key.strides()},
        value_strides_{value.strides()},
        key_cache_strides_{key_cache.strides()},
        value_cache_strides_{value_cache.strides()},
        slot_mapping_strides_{slot_mapping.strides()},
        k_scale_strides_{k_scale.strides()},
        v_scale_strides_{v_scale.strides()},
        key_type_{key.dtype()},
        key_cache_type_{key_cache.dtype()},
        value_cache_type_{value_cache.dtype()},
        k_scale_type_{k_scale.dtype()},
        v_scale_type_{v_scale.dtype()},
        kv_cache_dtype_{kv_cache_dtype},
        num_tokens_{slot_mapping.size(0)},
        num_heads_{key.size(1)},
        head_size_{key.size(2)},
        block_size_{key_cache.size(3)},
        x_{key_cache.size(4)},
        device_index_{key.device().index()} {
    assert(key.ndim() == 3 && value.ndim() == 3 &&
           "`ReshapeAndCache` requires 3D `key` and `value`");
    assert(key.shape() == value.shape() &&
           "`ReshapeAndCache` requires `key` and `value` to have the same "
           "shape");
    assert(key.dtype() == value.dtype() &&
           "`ReshapeAndCache` requires `key` and `value` to have the same "
           "dtype");
    assert((key_type_ == DataType::kFloat16 ||
            key_type_ == DataType::kBFloat16 ||
            key_type_ == DataType::kFloat32) &&
           "`ReshapeAndCache` supports float16, bfloat16, and float32 inputs");
    assert(key_cache.ndim() == 5 && value_cache.ndim() == 4 &&
           "`ReshapeAndCache` requires 5D `key_cache` and 4D `value_cache`");
    assert(slot_mapping.ndim() == 1 &&
           slot_mapping.dtype() == DataType::kInt64 &&
           slot_mapping.stride(0) == 1 &&
           "`ReshapeAndCache` requires contiguous int64 `slot_mapping`");
    assert(num_tokens_ <= key.size(0) &&
           "`ReshapeAndCache` requires enough key/value rows for all slots");
    assert(x_ > 0 && head_size_ % x_ == 0 &&
           "`ReshapeAndCache` requires `head_size` divisible by cache vector "
           "width");
    assert(key_cache.size(0) == value_cache.size(0) &&
           key_cache.size(1) == num_heads_ &&
           value_cache.size(1) == num_heads_ &&
           key_cache.size(2) == head_size_ / x_ &&
           value_cache.size(2) == head_size_ &&
           value_cache.size(3) == block_size_ &&
           "`ReshapeAndCache` cache shapes do not match key/value geometry");
    assert(key.stride(2) == 1 && key.stride(1) == head_size_ &&
           value.stride(2) == 1 && value.stride(1) == head_size_ &&
           "`ReshapeAndCache` requires contiguous head dimensions");
    assert(key_cache.stride(4) == 1 && key_cache.stride(3) == x_ &&
           key_cache.stride(2) == block_size_ * x_ &&
           key_cache.stride(1) == head_size_ * block_size_ &&
           key_cache.stride(0) == num_heads_ * head_size_ * block_size_ &&
           "`ReshapeAndCache` requires contiguous vLLM key-cache layout");
    assert(value_cache.stride(3) == 1 && value_cache.stride(2) == block_size_ &&
           value_cache.stride(1) == head_size_ * block_size_ &&
           value_cache.stride(0) == num_heads_ * head_size_ * block_size_ &&
           "`ReshapeAndCache` requires contiguous vLLM value-cache layout");
    assert((kv_cache_dtype_ == "auto" || kv_cache_dtype_ == "fp8" ||
            kv_cache_dtype_ == "fp8_e4m3" || kv_cache_dtype_ == "fp8_e5m2") &&
           "`ReshapeAndCache` received unsupported `kv_cache_dtype`");

    const bool quantized = kv_cache_dtype_ != "auto";
    assert((quantized ? key_cache_type_ == DataType::kUInt8
                      : key_cache_type_ == key_type_) &&
           key_cache_type_ == value_cache_type_ &&
           "`ReshapeAndCache` cache storage dtype does not match "
           "`kv_cache_dtype`");
    assert(k_scale.numel() == 1 && v_scale.numel() == 1 &&
           k_scale_type_ == DataType::kFloat32 &&
           v_scale_type_ == DataType::kFloat32 &&
           "`ReshapeAndCache` requires scalar float32 scales");
  }

  virtual void operator()(const Tensor key, const Tensor value,
                          Tensor key_cache, Tensor value_cache,
                          const Tensor slot_mapping, const Tensor k_scale,
                          const Tensor v_scale,
                          const std::string kv_cache_dtype) const = 0;

 protected:
  Tensor::Shape key_shape_;

  Tensor::Shape value_shape_;

  Tensor::Shape key_cache_shape_;

  Tensor::Shape value_cache_shape_;

  Tensor::Shape slot_mapping_shape_;

  Tensor::Shape k_scale_shape_;

  Tensor::Shape v_scale_shape_;

  Tensor::Strides key_strides_;

  Tensor::Strides value_strides_;

  Tensor::Strides key_cache_strides_;

  Tensor::Strides value_cache_strides_;

  Tensor::Strides slot_mapping_strides_;

  Tensor::Strides k_scale_strides_;

  Tensor::Strides v_scale_strides_;

  DataType key_type_;

  DataType key_cache_type_;

  DataType value_cache_type_;

  DataType k_scale_type_;

  DataType v_scale_type_;

  std::string kv_cache_dtype_;

  Tensor::Size num_tokens_{0};

  Tensor::Size num_heads_{0};

  Tensor::Size head_size_{0};

  Tensor::Size block_size_{0};

  Tensor::Size x_{0};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
