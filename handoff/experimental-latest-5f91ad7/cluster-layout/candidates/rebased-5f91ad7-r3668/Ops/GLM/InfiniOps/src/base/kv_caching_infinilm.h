#ifndef INFINI_OPS_BASE_KV_CACHING_INFINILM_H_
#define INFINI_OPS_BASE_KV_CACHING_INFINILM_H_

#include <cassert>

#include "operator.h"

namespace infini::ops {

/// \deprecated Migrate to an open-source-aligned operator when available. This
/// interface will be removed in a future release.
class [[deprecated(
    "Migrate to an open-source-aligned operator when available.")]]
KvCachingInfinilm : public Operator<KvCachingInfinilm> {
 public:
  KvCachingInfinilm(const Tensor k, const Tensor v,
                    const Tensor past_kv_lengths, Tensor k_cache,
                    Tensor v_cache)
      : k_cache_shape_{k_cache.shape()},
        k_cache_strides_{k_cache.strides()},
        v_cache_shape_{v_cache.shape()},
        v_cache_strides_{v_cache.strides()},
        k_shape_{k.shape()},
        k_strides_{k.strides()},
        v_shape_{v.shape()},
        v_strides_{v.strides()},
        past_kv_lengths_shape_{past_kv_lengths.shape()},
        data_type_{k_cache.dtype()},
        past_kv_lengths_type_{past_kv_lengths.dtype()},
        batch_size_{k_cache.size(0)},
        num_kv_heads_{k_cache.size(1)},
        max_seq_len_{k_cache.size(2)},
        seq_len_{k.size(2)},
        hidden_size_{k_cache.size(3)},
        output_size_{k.numel()},
        device_index_{k_cache.device().index()} {
    assert(k_cache.ndim() == 4 && v_cache.ndim() == 4 && k.ndim() == 4 &&
           v.ndim() == 4 && "`KvCachingInfinilm` tensors must be 4D");
    assert(k_cache_shape_ == v_cache_shape_ &&
           "`KvCachingInfinilm` cache shapes must match");
    assert(k_shape_ == v_shape_ &&
           "`KvCachingInfinilm` source shapes must match");
    assert(k.size(0) == batch_size_ && k.size(1) == num_kv_heads_ &&
           k.size(3) == hidden_size_ &&
           "`KvCachingInfinilm` source shape must match cache "
           "batch/head/hidden dims");
    assert(seq_len_ <= max_seq_len_ &&
           "`KvCachingInfinilm` source sequence length exceeds cache length");
    assert(k_cache.dtype() == v_cache.dtype() && k_cache.dtype() == k.dtype() &&
           k_cache.dtype() == v.dtype() &&
           "`KvCachingInfinilm` K/V tensors must have the same dtype");
    assert(
        (data_type_ == DataType::kFloat16 ||
         data_type_ == DataType::kBFloat16 ||
         data_type_ == DataType::kFloat32) &&
        "`KvCachingInfinilm` K/V dtype must be float16, bfloat16, or float32");
    assert((past_kv_lengths_type_ == DataType::kInt32 ||
            past_kv_lengths_type_ == DataType::kInt64) &&
           "`KvCachingInfinilm` past_kv_lengths dtype must be int32 or int64");
    assert(past_kv_lengths.ndim() == 1 &&
           past_kv_lengths.size(0) == batch_size_ &&
           "`KvCachingInfinilm` past_kv_lengths shape must be (batch_size,)");
    assert(!k_cache.HasBroadcastDim() && !v_cache.HasBroadcastDim() &&
           "`KvCachingInfinilm` caches must not have broadcasted dimensions");
  }

  virtual void operator()(const Tensor k, const Tensor v,
                          const Tensor past_kv_lengths, Tensor k_cache,
                          Tensor v_cache) const = 0;

 protected:
  Tensor::Shape k_cache_shape_;

  Tensor::Strides k_cache_strides_;

  Tensor::Shape v_cache_shape_;

  Tensor::Strides v_cache_strides_;

  Tensor::Shape k_shape_;

  Tensor::Strides k_strides_;

  Tensor::Shape v_shape_;

  Tensor::Strides v_strides_;

  Tensor::Shape past_kv_lengths_shape_;

  DataType data_type_;

  DataType past_kv_lengths_type_;

  Tensor::Size batch_size_{0};

  Tensor::Size num_kv_heads_{0};

  Tensor::Size max_seq_len_{0};

  Tensor::Size seq_len_{0};

  Tensor::Size hidden_size_{0};

  Tensor::Size output_size_{0};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
