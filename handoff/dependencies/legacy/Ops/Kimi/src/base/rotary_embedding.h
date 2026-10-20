#ifndef INFINI_OPS_BASE_ROTARY_EMBEDDING_H_
#define INFINI_OPS_BASE_ROTARY_EMBEDDING_H_

#include <optional>

#include "data_type.h"
#include "operator.h"

namespace infini::ops {

class RotaryEmbedding : public Operator<RotaryEmbedding> {
 public:
  RotaryEmbedding(const Tensor positions, Tensor query,
                  std::optional<Tensor> key, const Tensor cos_sin_cache,
                  int64_t head_size, bool is_neox, int64_t rope_dim_offset = 0,
                  bool inverse = false)
      : positions_shape_{positions.shape()},
        query_shape_{query.shape()},
        key_shape_{key.has_value() ? Tensor::Shape{key->shape()}
                                   : Tensor::Shape{}},
        cos_sin_cache_shape_{cos_sin_cache.shape()},
        positions_strides_{positions.strides()},
        query_strides_{query.strides()},
        key_strides_{key.has_value() ? Tensor::Strides{key->strides()}
                                     : Tensor::Strides{}},
        cos_sin_cache_strides_{cos_sin_cache.strides()},
        positions_type_{positions.dtype()},
        query_type_{query.dtype()},
        cos_sin_cache_type_{cos_sin_cache.dtype()},
        num_tokens_{positions.numel()},
        positions_ndim_{positions.ndim()},
        head_size_{head_size},
        rot_dim_{static_cast<int64_t>(cos_sin_cache.size(1))},
        rope_dim_offset_{rope_dim_offset},
        is_neox_{is_neox},
        inverse_{inverse},
        query_hidden_size_{num_tokens_ == 0 ? 0 : query.numel() / num_tokens_},
        key_hidden_size_{key.has_value() && num_tokens_ != 0
                             ? key->numel() / num_tokens_
                             : 0},
        num_heads_{head_size_ == 0 ? 0 : query_hidden_size_ / head_size_},
        num_kv_heads_{head_size_ == 0
                          ? 0
                          : (key.has_value() ? key_hidden_size_ / head_size_
                                             : num_heads_)},
        query_token_stride_{query.stride(positions_ndim_ - 1)},
        key_token_stride_{key.has_value() ? key->stride(positions_ndim_ - 1)
                                          : 0},
        query_head_stride_{query.ndim() == positions_ndim_ + 2
                               ? query.stride(-2)
                               : head_size_},
        key_head_stride_{key.has_value() && key->ndim() == positions_ndim_ + 2
                             ? key->stride(-2)
                             : head_size_},
        device_index_{query.device().index()} {
    assert((positions_ndim_ == 1 || positions_ndim_ == 2) &&
           "`RotaryEmbedding` requires 1D or 2D `positions`");
    assert(positions_type_ == DataType::kInt64 &&
           "`RotaryEmbedding` requires int64 `positions`");
    assert(positions.stride(-1) == 1 &&
           (positions_ndim_ == 1 || positions.stride(0) == positions.size(1)) &&
           "`RotaryEmbedding` requires contiguous `positions`");
    assert((query.ndim() == positions_ndim_ + 1 ||
            query.ndim() == positions_ndim_ + 2) &&
           "`RotaryEmbedding` received unsupported `query` rank");
    assert(head_size_ > 0 && query_hidden_size_ % head_size_ == 0 &&
           "`RotaryEmbedding` requires query hidden size divisible by "
           "`head_size`");
    assert((query_type_ == DataType::kFloat16 ||
            query_type_ == DataType::kBFloat16 ||
            query_type_ == DataType::kFloat32) &&
           "`RotaryEmbedding` supports float16, bfloat16, and float32 query");
    assert(cos_sin_cache.ndim() == 2 && rot_dim_ % 2 == 0 &&
           cos_sin_cache.stride(1) == 1 &&
           "`RotaryEmbedding` requires contiguous 2D `cos_sin_cache` with "
           "even rotary dimension");
    assert((cos_sin_cache_type_ == DataType::kFloat16 ||
            cos_sin_cache_type_ == DataType::kBFloat16 ||
            cos_sin_cache_type_ == DataType::kFloat32) &&
           "`RotaryEmbedding` supports float16, bfloat16, and float32 cache");
    assert(rope_dim_offset_ >= 0 && rot_dim_ + rope_dim_offset_ <= head_size_ &&
           "`RotaryEmbedding` rotary dimensions exceed `head_size`");
    assert(query.stride(-1) == 1 &&
           "`RotaryEmbedding` requires contiguous query head dimensions");

    if (positions_ndim_ == 1) {
      assert(query.size(0) == positions.size(0) &&
             (!key.has_value() || key->size(0) == positions.size(0)) &&
             "`RotaryEmbedding` requires matching token counts");
    } else {
      assert(query.size(0) == positions.size(0) &&
             query.size(1) == positions.size(1) &&
             (!key.has_value() || (key->size(0) == positions.size(0) &&
                                   key->size(1) == positions.size(1))) &&
             "`RotaryEmbedding` requires matching batch and sequence sizes");
    }

    if (query.ndim() == positions_ndim_ + 2) {
      assert(query.size(-1) == static_cast<Tensor::Size>(head_size_) &&
             "`RotaryEmbedding` query head dimension does not match "
             "`head_size`");
    }

    if (key.has_value()) {
      assert((key->ndim() == positions_ndim_ + 1 ||
              key->ndim() == positions_ndim_ + 2) &&
             key_hidden_size_ % head_size_ == 0 &&
             key->dtype() == query_type_ && key->stride(-1) == 1 &&
             "`RotaryEmbedding` key layout or dtype is incompatible with "
             "query");
      assert(num_kv_heads_ > 0 && num_heads_ % num_kv_heads_ == 0 &&
             "`RotaryEmbedding` requires query heads divisible by key heads");
      if (key->ndim() == positions_ndim_ + 2) {
        assert(key->size(-1) == static_cast<Tensor::Size>(head_size_) &&
               "`RotaryEmbedding` key head dimension does not match "
               "`head_size`");
      }
    }
  }

  virtual void operator()(const Tensor positions, Tensor query,
                          std::optional<Tensor> key, const Tensor cos_sin_cache,
                          int64_t head_size, bool is_neox,
                          int64_t rope_dim_offset = 0,
                          bool inverse = false) const = 0;

 protected:
  Tensor::Shape positions_shape_;

  Tensor::Shape query_shape_;

  Tensor::Shape key_shape_;

  Tensor::Shape cos_sin_cache_shape_;

  Tensor::Strides positions_strides_;

  Tensor::Strides query_strides_;

  Tensor::Strides key_strides_;

  Tensor::Strides cos_sin_cache_strides_;

  DataType positions_type_;

  DataType query_type_;

  DataType cos_sin_cache_type_;

  Tensor::Size num_tokens_{0};

  Tensor::Size positions_ndim_{0};

  int64_t head_size_{0};

  int64_t rot_dim_{0};

  int64_t rope_dim_offset_{0};

  bool is_neox_{false};

  bool inverse_{false};

  Tensor::Size query_hidden_size_{0};

  Tensor::Size key_hidden_size_{0};

  Tensor::Size num_heads_{0};

  Tensor::Size num_kv_heads_{0};

  int64_t query_token_stride_{0};

  int64_t key_token_stride_{0};

  int64_t query_head_stride_{0};

  int64_t key_head_stride_{0};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
