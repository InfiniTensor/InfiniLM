#include "native/cuda/nvidia/ops/take/kernel.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace infini::ops::nvidia_take_detail {
namespace {

template <typename Bits>
__global__ void TakeKernel(const Bits* input, const std::int64_t* index,
                           Bits* out, std::int64_t input_count,
                           std::size_t output_count) {
  const auto step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (auto pos = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       pos < output_count; pos += step) {
    auto source = index[pos];
    if (source < -input_count || source >= input_count) {
      // Unlike a silent skip, an unconditional trap remains active in release
      // builds. Callers must observe stream/graph completion; the CUDA context
      // is poisoned after this error, so negative tests need a child process.
      asm volatile("trap;");
      return;
    }
    if (source < 0) source += input_count;
    out[pos] = input[source];
  }
}

template <typename Bits>
void Submit(const void* input, const std::int64_t* index, void* out,
            std::size_t input_count, std::size_t output_count,
            cudaStream_t stream) {
  constexpr unsigned kThreads = 256;
  const auto blocks = static_cast<unsigned>(
      std::min<std::size_t>((output_count - 1) / kThreads + 1, 65535));
  TakeKernel<Bits><<<blocks, kThreads, 0, stream>>>(
      static_cast<const Bits*>(input), index, static_cast<Bits*>(out),
      static_cast<std::int64_t>(input_count), output_count);
}

}  // namespace

void Launch(const void* input, const std::int64_t* index, void* out,
            std::size_t input_count, std::size_t output_count,
            std::size_t element_size, cudaStream_t stream) {
  switch (element_size) {
    case 1: Submit<std::uint8_t>(input, index, out, input_count, output_count, stream); break;
    case 2: Submit<std::uint16_t>(input, index, out, input_count, output_count, stream); break;
    case 4: Submit<std::uint32_t>(input, index, out, input_count, output_count, stream); break;
    case 8: Submit<std::uint64_t>(input, index, out, input_count, output_count, stream); break;
    default: throw std::invalid_argument("NVIDIA Take unsupported element size");
  }
  const auto status = cudaPeekAtLastError();
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

}  // namespace infini::ops::nvidia_take_detail
