#ifndef INFINI_OPS_CUDA_KV_CACHING_INFINILM_KERNEL_H_
#define INFINI_OPS_CUDA_KV_CACHING_INFINILM_KERNEL_H_

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>

#include "base/kv_caching_infinilm.h"
#include "common/generic_utils.h"
#include "data_type.h"
#include "dispatcher.h"
#include "native/cuda/ops/kv_caching_infinilm/kernel.cuh"
#include "native/cuda/runtime_utils.h"

namespace infini::ops {

template <typename Backend>
class CudaKvCachingInfinilm : public KvCachingInfinilm {
 public:
  CudaKvCachingInfinilm(const Tensor k, const Tensor v,
                        const Tensor past_kv_lengths, Tensor k_cache,
                        Tensor v_cache)
      : KvCachingInfinilm{k, v, past_kv_lengths, k_cache, v_cache} {
    constexpr size_t ndim = 4;
    size_t strides_size = ndim * sizeof(*d_k_cache_strides_);
    const size_t metadata_size = 4 * strides_size;
    std::vector<std::byte> metadata(metadata_size);

    Backend::Malloc((void**)&d_metadata_, metadata_size);

    size_t offset = 0;
    d_k_cache_strides_ =
        reinterpret_cast<Tensor::Stride*>(d_metadata_ + offset);
    std::memcpy(metadata.data() + offset, k_cache_strides_.data(),
                strides_size);
    offset += strides_size;

    d_v_cache_strides_ =
        reinterpret_cast<Tensor::Stride*>(d_metadata_ + offset);
    std::memcpy(metadata.data() + offset, v_cache_strides_.data(),
                strides_size);
    offset += strides_size;

    d_k_strides_ = reinterpret_cast<Tensor::Stride*>(d_metadata_ + offset);
    std::memcpy(metadata.data() + offset, k_strides_.data(), strides_size);
    offset += strides_size;

    d_v_strides_ = reinterpret_cast<Tensor::Stride*>(d_metadata_ + offset);
    std::memcpy(metadata.data() + offset, v_strides_.data(), strides_size);

    Backend::Memcpy(d_metadata_, metadata.data(), metadata_size,
                    Backend::kMemcpyHostToDevice);
  }

  ~CudaKvCachingInfinilm() { Backend::Free(d_metadata_); }

  void operator()(const Tensor k, const Tensor v, const Tensor past_kv_lengths,
                  Tensor k_cache, Tensor v_cache) const override {
    auto cuda_stream =
        static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);
    int block_size = std::min(
        RuntimeUtils<Backend::kDeviceType>::GetOptimalBlockSize(), 1024);
    dim3 block(std::min(static_cast<Tensor::Size>(block_size), output_size_));
    dim3 grid(utils::CeilDiv(output_size_, block.x));

    using IndexTypes = List<DataType::kInt32, DataType::kInt64>;
    DispatchFunc<AllFloatTypes, IndexTypes, List<128, 256, 512, 1024>>(
        {static_cast<int64_t>(data_type_),
         static_cast<int64_t>(past_kv_lengths_type_), block_size},
        [&](auto list_tag) {
          using T = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          using TIndex =
              TypeMapType<Backend::kDeviceType, ListGet<1>(list_tag)>;
          constexpr int kBlockSize = ListGet<2>(list_tag);

          KvCachingInfinilmKernel<T, TIndex, kBlockSize>
              <<<grid, block, 0, cuda_stream>>>(
                  reinterpret_cast<T*>(k_cache.data()),
                  reinterpret_cast<T*>(v_cache.data()),
                  reinterpret_cast<const T*>(k.data()),
                  reinterpret_cast<const T*>(v.data()),
                  reinterpret_cast<const TIndex*>(past_kv_lengths.data()),
                  d_k_cache_strides_, d_v_cache_strides_, d_k_strides_,
                  d_v_strides_, output_size_, num_kv_heads_, seq_len_,
                  hidden_size_);
        },
        "CudaKvCachingInfinilm::operator()");
  }

 private:
  std::byte* d_metadata_{nullptr};

  Tensor::Stride* d_k_cache_strides_{nullptr};

  Tensor::Stride* d_v_cache_strides_{nullptr};

  Tensor::Stride* d_k_strides_{nullptr};

  Tensor::Stride* d_v_strides_{nullptr};
};

}  // namespace infini::ops

#endif
