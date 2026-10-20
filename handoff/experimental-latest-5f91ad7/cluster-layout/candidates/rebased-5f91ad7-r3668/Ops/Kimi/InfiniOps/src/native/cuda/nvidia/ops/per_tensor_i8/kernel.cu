#include "native/cuda/nvidia/ops/per_tensor_i8/kernel.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <stdexcept>
namespace infini::ops::nvidia_per_tensor_i8_detail {
namespace {
__device__ std::int64_t Offset(std::size_t linear, const Layout& l,
                               bool packed) {
  std::int64_t offset = 0;
  for (std::int64_t d = l.ndim; d-- > 0;) {
    auto coord = linear % l.shape[d]; linear /= l.shape[d];
    offset += coord * (packed ? l.packed_stride[d] : l.x_stride[d]);
  }
  return offset;
}
__device__ void AtomicMax(float* ptr, float value) {
  auto bits = reinterpret_cast<int*>(ptr);
  int old = *bits, assumed;
  do {
    assumed = old;
    old = atomicCAS(bits, assumed,
                    __float_as_int(fmaxf(value, __int_as_float(assumed))));
  } while (old != assumed);
}
template <typename T> __global__ void Absmax(const T* x, float* scale,
                                             Layout l, std::size_t count) {
  float maximum = 0;
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += std::size_t(gridDim.x) * blockDim.x)
    maximum = fmaxf(maximum, fabsf(float(x[Offset(i, l, false)])));
  // Preserve the original dynamic contract: atomic max with caller-provided
  // scale, without epsilon or clearing scale between recorded executions.
  AtomicMax(scale, maximum / 127.0f);
}
template <typename T> __global__ void Quant(const T* x, std::int8_t* packed,
    const float* scale, Layout l, std::size_t count) {
  const float inverse = 1.0f / scale[0];
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += std::size_t(gridDim.x) * blockDim.x) {
    const float value = float(x[Offset(i, l, false)]) * inverse;
    const float rounded = floorf(fabsf(value) + 0.5f);
    int q = value >= 0 ? int(rounded) : -int(rounded);
    q = q > 127 ? 127 : (q < -127 ? -127 : q);
    packed[Offset(i, l, true)] = std::int8_t(q);
  }
}
template <typename T> __global__ void Dequant(T* x, const std::int8_t* packed,
    const float* scale, Layout l, std::size_t count) {
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += std::size_t(gridDim.x) * blockDim.x)
    x[Offset(i, l, false)] = T(float(packed[Offset(i, l, true)]) * scale[0]);
}
void CheckLaunch() {
  const auto error = cudaGetLastError();
  if (error != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(error));
}
template <typename T> void SubmitQuant(const void* x, std::int8_t* packed,
    float* scale, Layout l, std::size_t count, bool is_static, cudaStream_t stream) {
  const unsigned blocks = unsigned(std::min<std::size_t>((count-1)/256+1,65535));
  if (!is_static) {
    Absmax<T><<<blocks,256,0,stream>>>(static_cast<const T*>(x),scale,l,count);
    CheckLaunch();
  }
  Quant<T><<<blocks,256,0,stream>>>(static_cast<const T*>(x),packed,scale,l,count);
  CheckLaunch();
}
template <typename T> void SubmitDequant(void* x, const std::int8_t* packed,
    const float* scale, Layout l, std::size_t count, cudaStream_t stream) {
  const unsigned blocks = unsigned(std::min<std::size_t>((count-1)/256+1,65535));
  Dequant<T><<<blocks,256,0,stream>>>(static_cast<T*>(x),packed,scale,l,count);
  CheckLaunch();
}
}
void Quantize(const void* x, std::int8_t* packed, float* scale, DataType dtype,
              Layout l, std::size_t count, bool is_static, cudaStream_t stream) {
  if (!count) return;
  switch (dtype) {
    case DataType::kFloat16: SubmitQuant<__half>(x,packed,scale,l,count,is_static,stream); break;
    case DataType::kBFloat16: SubmitQuant<__nv_bfloat16>(x,packed,scale,l,count,is_static,stream); break;
    case DataType::kFloat32: SubmitQuant<float>(x,packed,scale,l,count,is_static,stream); break;
    default: throw std::invalid_argument("unsupported per-tensor INT8 dtype");
  }
}
void Dequantize(void* x, const std::int8_t* packed, const float* scale,
                DataType dtype, Layout l, std::size_t count, cudaStream_t stream) {
  if (!count) return;
  switch (dtype) {
    case DataType::kFloat16: SubmitDequant<__half>(x,packed,scale,l,count,stream); break;
    case DataType::kBFloat16: SubmitDequant<__nv_bfloat16>(x,packed,scale,l,count,stream); break;
    case DataType::kFloat32: SubmitDequant<float>(x,packed,scale,l,count,stream); break;
    default: throw std::invalid_argument("unsupported per-tensor INT8 dtype");
  }
}
}
