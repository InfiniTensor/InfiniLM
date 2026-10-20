#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <stdexcept>

#include "native/cuda/nvidia/ops/pad/kernel.h"

namespace infini::ops::nvidia_pad_detail {
namespace {
template <typename T>
__global__ void PadKernel(const T* input, T* out, Layout layout, T fill,
                          std::size_t count) {
  const auto start =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t linear = start; linear < count; linear += step) {
    std::size_t remaining = linear;
    std::int64_t input_offset = 0, output_offset = 0;
    bool inside = true;
    for (std::int64_t d = layout.ndim; d-- > 0;) {
      const auto coord = static_cast<std::int64_t>(
          remaining % static_cast<std::size_t>(layout.output_shape[d]));
      remaining /= static_cast<std::size_t>(layout.output_shape[d]);
      output_offset += coord * layout.output_strides[d];
      const auto source = coord - layout.left_padding[d];
      if (source < 0 || source >= layout.input_shape[d])
        inside = false;
      else
        input_offset += source * layout.input_strides[d];
    }
    out[output_offset] = inside ? input[input_offset] : fill;
  }
}

template <typename T>
void Submit(const void* input, void* out, Layout layout, T value,
            std::size_t count, cudaStream_t stream) {
  constexpr unsigned kThreads = 256;
  const auto blocks = static_cast<unsigned>(
      std::min<std::size_t>((count - 1) / kThreads + 1, 65535));
  PadKernel<T><<<blocks, kThreads, 0, stream>>>(
      static_cast<const T*>(input), static_cast<T*>(out), layout, value, count);
}
}  // namespace

void Launch(const void* input, void* out, DataType dtype, double value,
            Layout layout, std::size_t output_count, cudaStream_t stream) {
  if (!output_count) return;
  switch (dtype) {
    case DataType::kFloat16:
      Submit<__half>(input, out, layout,
                     __float2half_rn(static_cast<float>(value)), output_count,
                     stream);
      break;
    case DataType::kBFloat16:
      Submit<__nv_bfloat16>(input, out, layout,
                            __float2bfloat16_rn(static_cast<float>(value)),
                            output_count, stream);
      break;
    case DataType::kFloat32:
      Submit<float>(input, out, layout, static_cast<float>(value), output_count,
                    stream);
      break;
    case DataType::kFloat64:
      Submit<double>(input, out, layout, value, output_count, stream);
      break;
    default:
      throw std::invalid_argument("NVIDIA Pad unsupported dtype");
  }
  const auto status = cudaPeekAtLastError();
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
}  // namespace infini::ops::nvidia_pad_detail
