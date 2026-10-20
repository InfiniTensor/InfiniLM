#include "native/cuda/hygon/ops/layer_norm/kernel.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>

#include <stdexcept>

namespace infini::ops::hygon_layer_norm_detail {
namespace {
constexpr unsigned kThreads = 256;

template <typename T>
__global__ void LayerNormKernel(const T* input, const T* weight, const T* bias,
                                T* out, Layout layout, float eps) {
  std::size_t row = blockIdx.x;
  std::size_t input_offset = 0;
  std::size_t out_offset = 0;
  for (unsigned i = layout.rank - 1; i > 0; --i) {
    const auto axis = i - 1;
    const auto coordinate = row % layout.shape[axis];
    row /= layout.shape[axis];
    input_offset += coordinate * layout.input_strides[axis];
    out_offset += coordinate * layout.out_strides[axis];
  }

  using Reduce = cub::BlockReduce<float, kThreads>;
  __shared__ typename Reduce::TempStorage reduce_storage;
  __shared__ float mean;
  __shared__ float inv_std;
  float partial = 0.0f;
  for (std::size_t d = threadIdx.x; d < layout.dim; d += kThreads)
    partial += static_cast<float>(input[input_offset +
        d * layout.input_strides[layout.rank - 1]]);
  const float sum = Reduce(reduce_storage).Sum(partial);
  if (threadIdx.x == 0) mean = sum / static_cast<float>(layout.dim);
  __syncthreads();

  partial = 0.0f;
  for (std::size_t d = threadIdx.x; d < layout.dim; d += kThreads) {
    const float centered = static_cast<float>(input[input_offset +
        d * layout.input_strides[layout.rank - 1]]) - mean;
    partial += centered * centered;
  }
  const float variance_sum = Reduce(reduce_storage).Sum(partial);
  if (threadIdx.x == 0)
    inv_std = rsqrtf(variance_sum / static_cast<float>(layout.dim) + eps);
  __syncthreads();

  for (std::size_t d = threadIdx.x; d < layout.dim; d += kThreads) {
    const float x = static_cast<float>(input[input_offset +
        d * layout.input_strides[layout.rank - 1]]);
    const float scale = static_cast<float>(weight[d * layout.weight_stride]);
    const float shift = bias ? static_cast<float>(bias[d * layout.bias_stride]) : 0.0f;
    out[out_offset + d * layout.out_strides[layout.rank - 1]] =
        static_cast<T>((x - mean) * inv_std * scale + shift);
  }
}

template <typename T>
void Submit(const void* input, const void* weight, const void* bias, void* out,
            Layout layout, std::size_t rows, float eps, cudaStream_t stream) {
  LayerNormKernel<T><<<static_cast<unsigned>(rows), kThreads, 0, stream>>>(
      static_cast<const T*>(input), static_cast<const T*>(weight),
      static_cast<const T*>(bias), static_cast<T*>(out), layout, eps);
}
}  // namespace

void Launch(const void* input, const void* weight, const void* bias, void* out,
            DataType dtype, Layout layout, std::size_t rows,
            float eps, cudaStream_t stream) {
  switch (dtype) {
    case DataType::kFloat16:
      Submit<__half>(input, weight, bias, out, layout, rows, eps, stream);
      break;
    case DataType::kBFloat16:
      Submit<__nv_bfloat16>(input, weight, bias, out, layout, rows, eps, stream);
      break;
    case DataType::kFloat32:
      Submit<float>(input, weight, bias, out, layout, rows, eps, stream);
      break;
    default:
      throw std::invalid_argument("HYGON LayerNorm unsupported dtype");
  }
  const auto status = cudaPeekAtLastError();
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
}  // namespace infini::ops::hygon_layer_norm_detail
