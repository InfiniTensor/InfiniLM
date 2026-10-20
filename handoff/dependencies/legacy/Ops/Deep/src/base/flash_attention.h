#ifndef INFINI_OPS_BASE_FLASH_ATTENTION_H_
#define INFINI_OPS_BASE_FLASH_ATTENTION_H_

#include <cstddef>
#include <optional>
#include <vector>

#include "operator.h"

namespace infini::ops {

// Legacy fused multi-head / grouped-query attention interface.
// Layout: `query` / `key` / `value` are `[T, N, D]` (TND).
// Prefill uses `cu_seqlens_q` / `cu_seqlens_kv` for variable-length packing.
// Decode uses `block_table` for paged KV cache lookup.
/// \deprecated Use `ScaledDotProductAttention` for standard attention
/// semantics. This interface will be removed in a future release.
class [[deprecated(
    "Use `ScaledDotProductAttention` for standard attention "
    "semantics.")]] FlashAttention : public Operator<FlashAttention> {
 public:
  FlashAttention(const Tensor query, const Tensor key, const Tensor value,
                 std::optional<Tensor> cu_seqlens_q,
                 std::optional<Tensor> cu_seqlens_kv,
                 std::optional<Tensor> block_table, int64_t num_heads,
                 int64_t num_kv_heads, int64_t head_size, double scale,
                 bool causal, int64_t window_left, int64_t window_right,
                 int64_t block_size, Tensor output)
      : num_tokens_{query.size(0)},
        num_heads_{num_heads},
        num_kv_heads_{num_kv_heads},
        head_size_{head_size},
        scale_{scale},
        causal_{causal},
        window_left_{window_left},
        window_right_{window_right},
        block_size_{block_size},
        dtype_{query.dtype()},
        query_shape_{query.shape()},
        key_shape_{key.shape()},
        value_shape_{value.shape()},
        output_shape_{output.shape()},
        query_strides_{query.strides()},
        key_strides_{key.strides()},
        value_strides_{value.strides()},
        output_strides_{output.strides()},
        has_cu_seqlens_q_{cu_seqlens_q.has_value()},
        has_cu_seqlens_kv_{cu_seqlens_kv.has_value()},
        has_block_table_{block_table.has_value()} {
    assert(num_heads % num_kv_heads == 0 &&
           "`FlashAttention` requires num_heads divisible by num_kv_heads");
    assert(query.ndim() == 3 &&
           "`FlashAttention` requires query to be 3D [T, N, D]");
  }

  virtual void operator()(const Tensor query, const Tensor key,
                          const Tensor value,
                          std::optional<Tensor> cu_seqlens_q,
                          std::optional<Tensor> cu_seqlens_kv,
                          std::optional<Tensor> block_table, int64_t num_heads,
                          int64_t num_kv_heads, int64_t head_size, double scale,
                          bool causal, int64_t window_left,
                          int64_t window_right, int64_t block_size,
                          Tensor output) const = 0;

 protected:
  Tensor::Size num_tokens_{0};

  int64_t num_heads_{0};

  int64_t num_kv_heads_{0};

  int64_t head_size_{0};

  double scale_{0.0};

  bool causal_{false};

  int64_t window_left_{-1};

  int64_t window_right_{-1};

  int64_t block_size_{0};

  const DataType dtype_;

  Tensor::Shape query_shape_;

  Tensor::Shape key_shape_;

  Tensor::Shape value_shape_;

  Tensor::Shape output_shape_;

  Tensor::Strides query_strides_;

  Tensor::Strides key_strides_;

  Tensor::Strides value_strides_;

  Tensor::Strides output_strides_;

  bool has_cu_seqlens_q_{false};

  bool has_cu_seqlens_kv_{false};

  bool has_block_table_{false};
};

}  // namespace infini::ops

#endif
