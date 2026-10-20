#ifndef INFINI_OPS_BASE_PAGED_CACHING_INFINILM_H_
#define INFINI_OPS_BASE_PAGED_CACHING_INFINILM_H_

#include <cassert>
#include <cstddef>

#include "data_type.h"
#include "operator.h"
#include "tensor.h"

namespace infini::ops {

/// \deprecated Migrate to an open-source-aligned operator when available. This
/// interface will be removed in a future release.
class [[deprecated(
    "Migrate to an open-source-aligned operator when available.")]]
PagedCachingInfinilm : public Operator<PagedCachingInfinilm> {
 public:
  PagedCachingInfinilm(const Tensor k, const Tensor v,
                       const Tensor slot_mapping, Tensor k_cache,
                       Tensor v_cache)
      : dtype_{k.dtype()},
        num_tokens_{slot_mapping.size(0)},
        num_kv_heads_{k.size(1)},
        head_size_{k.size(2)},
        block_size_{k_cache.size(2)},
        k_src_stride_{k.stride(0)},
        v_src_stride_{v.stride(0)},
        k_cache_block_stride_{k_cache.stride(0)},
        v_cache_block_stride_{v_cache.stride(0)},
        k_cache_head_stride_{k_cache.stride(1)},
        v_cache_head_stride_{v_cache.stride(1)},
        k_cache_slot_stride_{k_cache.stride(2)},
        v_cache_slot_stride_{v_cache.stride(2)} {
    assert(k.ndim() == 3 && v.ndim() == 3 &&
           "`PagedCachingInfinilm` requires `k` and `v` to be 3D");
    assert(k_cache.ndim() == 4 && v_cache.ndim() == 4 &&
           "`PagedCachingInfinilm` requires 4D cache tensors");
    assert(slot_mapping.ndim() == 1 &&
           "`PagedCachingInfinilm` requires 1D slot mapping");
    assert((dtype_ == DataType::kFloat16 || dtype_ == DataType::kBFloat16 ||
            dtype_ == DataType::kFloat32) &&
           "`PagedCachingInfinilm` supports float16, bfloat16, and float32");
    assert(v.dtype() == dtype_ && k_cache.dtype() == dtype_ &&
           v_cache.dtype() == dtype_);
    assert(slot_mapping.dtype() == DataType::kInt64 &&
           "`PagedCachingInfinilm` requires int64 slot mapping");
    assert(k.shape() == v.shape());
    assert(k_cache.shape() == v_cache.shape());
    assert(k_cache.size(1) == num_kv_heads_ && k_cache.size(3) == head_size_);
    assert(k.stride(2) == 1 && v.stride(2) == 1);
    assert(k_cache.stride(3) == 1 && v_cache.stride(3) == 1);
  }

  virtual void operator()(const Tensor k, const Tensor v,
                          const Tensor slot_mapping, Tensor k_cache,
                          Tensor v_cache) const = 0;

 protected:
  DataType dtype_;

  std::size_t num_tokens_{0};

  std::size_t num_kv_heads_{0};

  std::size_t head_size_{0};

  std::size_t block_size_{0};

  Tensor::Stride k_src_stride_{0};

  Tensor::Stride v_src_stride_{0};

  Tensor::Stride k_cache_block_stride_{0};

  Tensor::Stride v_cache_block_stride_{0};

  Tensor::Stride k_cache_head_stride_{0};

  Tensor::Stride v_cache_head_stride_{0};

  Tensor::Stride k_cache_slot_stride_{0};

  Tensor::Stride v_cache_slot_stride_{0};
};

}  // namespace infini::ops

#endif
