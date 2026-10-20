#ifndef INFINI_OPS_CUDA_KERNEL_COMMONS_CUH_
#define INFINI_OPS_CUDA_KERNEL_COMMONS_CUH_

#include <type_traits>

#include "caster.h"

namespace infini::ops {

using AllCudaBlockSizes = List<128, 256, 512, 1024, 2048>;

template <typename Backend, typename = void>
struct BackendMaxBlockSize : std::integral_constant<int, 2048> {};

template <typename Backend>
struct BackendMaxBlockSize<Backend,
                           std::void_t<decltype(Backend::max_block_size)>>
    : std::integral_constant<int, Backend::max_block_size> {};

template <int max_block_size>
struct SupportedCudaBlockSizes;

template <>
struct SupportedCudaBlockSizes<2048> {
  using type = AllCudaBlockSizes;
};

template <>
struct SupportedCudaBlockSizes<1024> {
  using type = List<128, 256, 512, 1024>;
};

template <>
struct SupportedCudaBlockSizes<512> {
  using type = List<128, 256, 512>;
};

template <>
struct SupportedCudaBlockSizes<256> {
  using type = List<128, 256>;
};

template <>
struct SupportedCudaBlockSizes<128> {
  using type = List<128>;
};

template <int max_block_size>
using SupportedCudaBlockSizesType =
    typename SupportedCudaBlockSizes<max_block_size>::type;

__forceinline__ __device__ __host__ size_t
IndexToOffset(size_t flat_index, size_t ndim, const size_t* shape,
              const ptrdiff_t* strides) {
  size_t res = 0;
  for (size_t i = ndim; i-- > 0;) {
    res += (flat_index % shape[i]) * strides[i];
    flat_index /= shape[i];
  }
  return res;
}

// Selects the largest block size from `AllCudaBlockSizes` that does not exceed
// `max_threads_per_block`.
inline int ComputeOptimalBlockSize(int max_threads_per_block) {
  if (max_threads_per_block >= 2048) return 2048;
  if (max_threads_per_block >= 1024) return 1024;
  if (max_threads_per_block >= 512) return 512;
  if (max_threads_per_block >= 256) return 256;
  return 128;
}

}  // namespace infini::ops

#endif
