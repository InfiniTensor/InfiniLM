#include "native/cuda/nvidia/ops/sum/kernel.h"
#include "native/cuda/ops/sum/kernel.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>

namespace infini::ops::nvidia_sum_detail {
namespace {
template <typename T>
void Submit(const void* input, void* out, Layout layout,
            std::size_t output_count, cudaStream_t stream) {
  constexpr unsigned kThreads = 256;
  const auto blocks = static_cast<unsigned>(
      std::min<std::size_t>((output_count - 1) / kThreads + 1, 65535));
  SumKernel<T><<<blocks, kThreads, 0, stream>>>(
      static_cast<const T*>(input), static_cast<T*>(out), layout, output_count);
}
}  // namespace

void Launch(const void* input, void* out, DataType dtype, Layout layout,
            std::size_t output_count, cudaStream_t stream) {
  if (!output_count) return;
  switch (dtype) {
    case DataType::kFloat16:
      Submit<__half>(input, out, layout, output_count, stream);
      break;
    case DataType::kBFloat16:
      Submit<__nv_bfloat16>(input, out, layout, output_count, stream);
      break;
    case DataType::kFloat32:
      Submit<float>(input, out, layout, output_count, stream);
      break;
    default:
      throw std::invalid_argument("NVIDIA Sum unsupported dtype");
  }
  const auto status = cudaPeekAtLastError();
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
}  // namespace infini::ops::nvidia_sum_detail
