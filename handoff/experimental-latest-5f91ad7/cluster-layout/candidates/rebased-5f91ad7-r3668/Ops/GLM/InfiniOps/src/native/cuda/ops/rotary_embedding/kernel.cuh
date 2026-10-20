#ifndef INFINI_OPS_CUDA_ROTARY_EMBEDDING_KERNEL_CUH_
#define INFINI_OPS_CUDA_ROTARY_EMBEDDING_KERNEL_CUH_

#include <cstddef>
#include <cstdint>

#include "native/cuda/caster.cuh"

namespace infini::ops {

template <Device::Type kDev, typename T, typename TCache, bool kIsNeox>
__device__ __forceinline__ void ApplyRotaryPair(T* data, const TCache* cache,
                                                size_t rot_offset,
                                                size_t embed_dim,
                                                bool inverse) {
  const size_t x_index = kIsNeox ? rot_offset : 2 * rot_offset;
  const size_t y_index = kIsNeox ? embed_dim + rot_offset : 2 * rot_offset + 1;
  const size_t cache_index = kIsNeox ? x_index : x_index / 2;
  float cos_value = Caster<kDev>::template Cast<float>(cache[cache_index]);
  float sin_value =
      Caster<kDev>::template Cast<float>(cache[embed_dim + cache_index]);
  if (inverse) {
    sin_value = -sin_value;
  }

  const float x = Caster<kDev>::template Cast<float>(data[x_index]);
  const float y = Caster<kDev>::template Cast<float>(data[y_index]);
  data[x_index] = Caster<kDev>::template Cast<T>(x * cos_value - y * sin_value);
  data[y_index] = Caster<kDev>::template Cast<T>(y * cos_value + x * sin_value);
}

template <Device::Type kDev, typename T, typename TCache, bool kIsNeox>
__global__ void RotaryEmbeddingKernel(
    const int64_t* __restrict__ positions, T* query, T* key,
    const TCache* __restrict__ cos_sin_cache, int64_t cache_token_stride,
    int64_t query_token_stride, int64_t key_token_stride,
    int64_t query_head_stride, int64_t key_head_stride, size_t num_heads,
    size_t num_kv_heads, size_t rot_dim, size_t rope_dim_offset, bool inverse) {
  const size_t token_idx = blockIdx.x;
  const int64_t position = positions[token_idx];
  const TCache* cache = cos_sin_cache + position * cache_token_stride;
  const size_t embed_dim = rot_dim / 2;

  for (size_t i = threadIdx.x; i < num_heads * embed_dim; i += blockDim.x) {
    const size_t head_idx = i / embed_dim;
    T* data = query + token_idx * query_token_stride +
              head_idx * query_head_stride + rope_dim_offset;
    ApplyRotaryPair<kDev, T, TCache, kIsNeox>(data, cache, i % embed_dim,
                                              embed_dim, inverse);
  }

  if (key != nullptr) {
    for (size_t i = threadIdx.x; i < num_kv_heads * embed_dim;
         i += blockDim.x) {
      const size_t head_idx = i / embed_dim;
      T* data = key + token_idx * key_token_stride +
                head_idx * key_head_stride + rope_dim_offset;
      ApplyRotaryPair<kDev, T, TCache, kIsNeox>(data, cache, i % embed_dim,
                                                embed_dim, inverse);
    }
  }
}

}  // namespace infini::ops

#endif
