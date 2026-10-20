#ifndef INFINI_OPS_BASE_ROTARY_EMBEDDING_INFINILM_H_
#define INFINI_OPS_BASE_ROTARY_EMBEDDING_INFINILM_H_

#include <cassert>
#include <cstddef>

#include "data_type.h"
#include "operator.h"
#include "tensor.h"

namespace infini::ops {

/// \deprecated Use `RotaryEmbedding`. This interface will be removed in a
/// future release.
class [[deprecated("Use `RotaryEmbedding` instead.")]] RotaryEmbeddingInfinilm
    : public Operator<RotaryEmbeddingInfinilm> {
 public:
  RotaryEmbeddingInfinilm(const Tensor input, const Tensor pos_ids,
                          const Tensor sin_table, const Tensor cos_table,
                          bool is_neox, Tensor out)
      : ndim_{out.ndim()},
        batch_size_{ndim_ == 4 ? out.size(-4) : 1},
        seq_len_{out.size(-3)},
        nhead_{out.size(-2)},
        table_dim_{sin_table.size(1)},
        has_batch_dim_{ndim_ == 4},
        pos_has_batch_dim_{pos_ids.ndim() == 2},
        input_strides_{input.strides()},
        out_strides_{out.strides()},
        pos_strides_{pos_ids.strides()} {
    const auto head_dim = out.size(-1);
    const auto table_len = sin_table.size(0);
    const auto angle_dtype = sin_table.dtype();
    const auto pos_dtype = pos_ids.dtype();

    assert(input.shape() == out.shape() &&
           "`RotaryEmbeddingInfinilm` requires `input` and `out` same shape");
    assert(input.dtype() == out.dtype() &&
           "`RotaryEmbeddingInfinilm` requires `input` and `out` same dtype");
    assert((ndim_ == 3 || ndim_ == 4) &&
           "`RotaryEmbeddingInfinilm` requires 3D or 4D tensor");
    assert(head_dim % 2 == 0 &&
           "`RotaryEmbeddingInfinilm` requires head dimension to be even");
    assert(
        head_dim == table_dim_ * 2 &&
        "`RotaryEmbeddingInfinilm` requires table dim to be half of head dim");
    assert(pos_ids.ndim() == 1 || pos_ids.ndim() == 2);
    assert((pos_dtype == DataType::kInt32 || pos_dtype == DataType::kInt64) &&
           "`RotaryEmbeddingInfinilm` requires int32 or int64 position ids");
    assert(sin_table.shape() == cos_table.shape() &&
           "`RotaryEmbeddingInfinilm` requires sin_table and cos_table same "
           "shape");
    assert(sin_table.dtype() == cos_table.dtype() &&
           "`RotaryEmbeddingInfinilm` requires sin_table and cos_table same "
           "dtype");
    assert((angle_dtype == DataType::kFloat16 ||
            angle_dtype == DataType::kBFloat16 ||
            angle_dtype == DataType::kFloat32) &&
           "`RotaryEmbeddingInfinilm` requires float sin/cos tables");
    assert(sin_table.ndim() == 2 && cos_table.ndim() == 2 &&
           "`RotaryEmbeddingInfinilm` requires 2D sin/cos tables");
    assert(
        table_len >= seq_len_ &&
        "`RotaryEmbeddingInfinilm` requires table length >= sequence length");
    assert((pos_has_batch_dim_ ? (pos_ids.size(0) == batch_size_ &&
                                  pos_ids.size(1) == seq_len_)
                               : (pos_ids.size(0) == seq_len_)) &&
           "`RotaryEmbeddingInfinilm` requires pos_ids shape [seq] or [batch, "
           "seq]");
    assert(out_strides_[ndim_ - 1] == 1 && input_strides_[ndim_ - 1] == 1 &&
           "`RotaryEmbeddingInfinilm` requires contiguous head dimension");
    assert(sin_table.strides()[1] == 1 && cos_table.strides()[1] == 1 &&
           "`RotaryEmbeddingInfinilm` requires contiguous table dimension");
  }

  virtual void operator()(const Tensor input, const Tensor pos_ids,
                          const Tensor sin_table, const Tensor cos_table,
                          bool is_neox, Tensor out) const = 0;

 protected:
  Tensor::Size ndim_{0};

  Tensor::Size batch_size_{0};

  Tensor::Size seq_len_{0};

  Tensor::Size nhead_{0};

  Tensor::Size table_dim_{0};

  bool has_batch_dim_{false};

  bool pos_has_batch_dim_{false};

  Tensor::Strides input_strides_;

  Tensor::Strides out_strides_;

  Tensor::Strides pos_strides_;
};

}  // namespace infini::ops

#endif
