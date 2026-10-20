#include "native/cuda/nvidia/ops/cat/kernel.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace infini::ops::nvidia_cat_detail {
namespace {
__global__ void CatAxis0Bytes(const std::uint8_t* first,
                              const std::uint8_t* second, std::uint8_t* out,
                              std::size_t first_bytes,
                              std::size_t total_bytes) {
  auto offset = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (; offset < total_bytes; offset += step)
    out[offset] = offset < first_bytes ? first[offset]
                                       : second[offset - first_bytes];
}
}  // namespace

void LaunchAxis0Bytes(const void* first, const void* second, void* out,
                      std::size_t first_bytes, std::size_t total_bytes,
                      cudaStream_t stream) {
  constexpr unsigned kThreads = 256;
  const auto blocks = static_cast<unsigned>(
      std::min<std::size_t>((total_bytes - 1) / kThreads + 1, 65535));
  CatAxis0Bytes<<<blocks, kThreads, 0, stream>>>(
      static_cast<const std::uint8_t*>(first),
      static_cast<const std::uint8_t*>(second),
      static_cast<std::uint8_t*>(out), first_bytes, total_bytes);
  const auto status = cudaPeekAtLastError();
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
}  // namespace infini::ops::nvidia_cat_detail
