#ifndef INFINI_OPS_BASE_PAGED_ATTENTION_INFINILM_H_
#define INFINI_OPS_BASE_PAGED_ATTENTION_INFINILM_H_

#include <cassert>
#include <cstddef>
#include <optional>

#include "data_type.h"
#include "operator.h"
#include "tensor.h"

namespace infini::ops {

/// \deprecated Migrate to an open-source-aligned operator when available. This
/// interface will be removed in a future release.
class [[deprecated(
    "Migrate to an open-source-aligned operator when available.")]]
PagedAttentionInfinilm : public Operator<PagedAttentionInfinilm> {
 public:
  PagedAttentionInfinilm(const Tensor q, const Tensor k_cache,
                         const Tensor v_cache, const Tensor block_tables,
                         const Tensor seq_lens,
                         std::optional<Tensor> alibi_slopes, float scale,
                         Tensor out)
      : dtype_{q.dtype()},
        index_dtype_{block_tables.dtype()},
        scale_{scale},
        num_seqs_{q.size(0)},
        num_heads_{q.size(1)},
        num_kv_heads_{k_cache.size(1)},
        head_size_{q.size(2)},
        block_size_{k_cache.size(2)},
        max_num_blocks_per_seq_{block_tables.size(1)},
        q_stride_{q.stride(0)},
        q_head_stride_{q.stride(1)},
        k_cache_block_stride_{k_cache.stride(0)},
        k_cache_head_stride_{k_cache.stride(1)},
        k_cache_slot_stride_{k_cache.stride(2)},
        v_cache_block_stride_{v_cache.stride(0)},
        v_cache_head_stride_{v_cache.stride(1)},
        v_cache_slot_stride_{v_cache.stride(2)},
        out_stride_{out.stride(0)},
        out_head_stride_{out.stride(1)},
        block_table_batch_stride_{block_tables.stride(0)},
        seq_lens_stride_{seq_lens.stride(0)} {
    assert(q.ndim() == 3 && out.ndim() == 3);
    assert(k_cache.ndim() == 4 && v_cache.ndim() == 4);
    assert(block_tables.ndim() == 2 && seq_lens.ndim() == 1);
    assert((dtype_ == DataType::kFloat16 || dtype_ == DataType::kBFloat16) &&
           "`PagedAttentionInfinilm` supports float16 and bfloat16");
    assert(out.dtype() == dtype_ && k_cache.dtype() == dtype_ &&
           v_cache.dtype() == dtype_);
    assert(IsIndexDtype(index_dtype_) && seq_lens.dtype() == index_dtype_);
    assert(q.shape() == out.shape());
    assert(k_cache.shape() == v_cache.shape());
    assert(block_tables.size(0) == num_seqs_ && seq_lens.size(0) == num_seqs_);
    assert(k_cache.size(1) == num_kv_heads_ &&
           v_cache.size(1) == num_kv_heads_);
    assert(k_cache.size(3) == head_size_ && v_cache.size(3) == head_size_);
    assert((head_size_ == 64 || head_size_ == 128) &&
           "`PagedAttentionInfinilm` supports head sizes 64 and 128");
    assert(num_heads_ % num_kv_heads_ == 0);
    assert(q.stride(2) == 1 && out.stride(2) == 1);
    assert(k_cache.stride(3) == 1 && v_cache.stride(3) == 1);
    assert(block_tables.stride(1) == 1 && seq_lens.stride(0) == 1);
    assert(!alibi_slopes.has_value() ||
           (alibi_slopes->dtype() == DataType::kFloat32 &&
            alibi_slopes->ndim() == 1 && alibi_slopes->size(0) == num_heads_ &&
            alibi_slopes->stride(0) == 1));
  }

  virtual void operator()(const Tensor q, const Tensor k_cache,
                          const Tensor v_cache, const Tensor block_tables,
                          const Tensor seq_lens,
                          std::optional<Tensor> alibi_slopes, float scale,
                          Tensor out) const = 0;

 protected:
  static bool IsIndexDtype(DataType dtype) {
    return dtype == DataType::kInt32 || dtype == DataType::kInt64 ||
           dtype == DataType::kUInt32;
  }

  DataType dtype_;
  DataType index_dtype_;
  float scale_{1.0f};
  std::size_t num_seqs_{0};
  std::size_t num_heads_{0};
  std::size_t num_kv_heads_{0};
  std::size_t head_size_{0};
  std::size_t block_size_{0};
  std::size_t max_num_blocks_per_seq_{0};
  Tensor::Stride q_stride_{0};
  Tensor::Stride q_head_stride_{0};
  Tensor::Stride k_cache_block_stride_{0};
  Tensor::Stride k_cache_head_stride_{0};
  Tensor::Stride k_cache_slot_stride_{0};
  Tensor::Stride v_cache_block_stride_{0};
  Tensor::Stride v_cache_head_stride_{0};
  Tensor::Stride v_cache_slot_stride_{0};
  Tensor::Stride out_stride_{0};
  Tensor::Stride out_head_stride_{0};
  Tensor::Stride block_table_batch_stride_{0};
  Tensor::Stride seq_lens_stride_{0};
};

}  // namespace infini::ops

#endif
