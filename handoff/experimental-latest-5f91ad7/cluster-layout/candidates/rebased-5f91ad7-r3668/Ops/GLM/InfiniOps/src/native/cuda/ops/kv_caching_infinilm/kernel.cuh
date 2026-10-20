#ifndef INFINI_OPS_CUDA_KV_CACHING_INFINILM_KERNEL_CUH_
#define INFINI_OPS_CUDA_KV_CACHING_INFINILM_KERNEL_CUH_

#include <cstddef>
#include <cstdint>

namespace infini::ops {

template <typename T, typename TIndex, unsigned int block_size>
__global__ void KvCachingInfinilmKernel(
    T* __restrict__ k_cache, T* __restrict__ v_cache, const T* __restrict__ k,
    const T* __restrict__ v, const TIndex* __restrict__ past_kv_lengths,
    const ptrdiff_t* __restrict__ k_cache_strides,
    const ptrdiff_t* __restrict__ v_cache_strides,
    const ptrdiff_t* __restrict__ k_strides,
    const ptrdiff_t* __restrict__ v_strides, size_t output_size,
    size_t num_kv_heads, size_t seq_len, size_t hidden_size) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (size_t idx = tid; idx < output_size; idx += blockDim.x * gridDim.x) {
    size_t offset = idx;
    size_t d = offset % hidden_size;
    offset /= hidden_size;
    size_t s = offset % seq_len;
    offset /= seq_len;
    size_t h = offset % num_kv_heads;
    size_t b = offset / num_kv_heads;

    size_t cache_s = static_cast<size_t>(past_kv_lengths[b]) + s;
    ptrdiff_t k_cache_offset = b * k_cache_strides[0] + h * k_cache_strides[1] +
                               cache_s * k_cache_strides[2] +
                               d * k_cache_strides[3];
    ptrdiff_t v_cache_offset = b * v_cache_strides[0] + h * v_cache_strides[1] +
                               cache_s * v_cache_strides[2] +
                               d * v_cache_strides[3];
    ptrdiff_t k_offset = b * k_strides[0] + h * k_strides[1] +
                         s * k_strides[2] + d * k_strides[3];
    ptrdiff_t v_offset = b * v_strides[0] + h * v_strides[1] +
                         s * v_strides[2] + d * v_strides[3];

    k_cache[k_cache_offset] = k[k_offset];
    v_cache[v_cache_offset] = v[v_offset];
  }
}

}  // namespace infini::ops

#endif
