#ifndef INFINI_OPS_CUDA_CONVOLUTION_KERNEL_CUH_
#define INFINI_OPS_CUDA_CONVOLUTION_KERNEL_CUH_

#include <cstddef>

#include "native/cuda/caster.cuh"
#include "native/cuda/kernel_commons.cuh"

namespace infini::ops {

namespace {

__device__ __forceinline__ size_t
OffsetFromCoordinates(const size_t* __restrict__ coords, size_t ndim,
                      const ptrdiff_t* __restrict__ strides) {
  size_t offset = 0;
  for (size_t i = 0; i < ndim; ++i) {
    offset += coords[i] * strides[i];
  }
  return offset;
}

}  // namespace

template <Device::Type kDev, typename T, unsigned int block_size>
__global__ void ConvKernel(T* __restrict__ out, const T* __restrict__ input,
                           const T* __restrict__ weight,
                           const T* __restrict__ bias,
                           const size_t* __restrict__ input_shape,
                           const size_t* __restrict__ weight_shape,
                           const size_t* __restrict__ out_shape,
                           const ptrdiff_t* __restrict__ input_strides,
                           const ptrdiff_t* __restrict__ weight_strides,
                           const ptrdiff_t* __restrict__ out_strides,
                           const ptrdiff_t* __restrict__ bias_strides,
                           const int64_t* __restrict__ padding,
                           const int64_t* __restrict__ stride,
                           const int64_t* __restrict__ dilation,
                           size_t output_size, size_t spatial_ndim,
                           size_t kernel_size, int64_t groups, bool has_bias) {
  size_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear >= output_size) {
    return;
  }

  size_t coords[5] = {0, 0, 0, 0, 0};
  size_t tmp = linear;
  size_t ndim = spatial_ndim + 2;
  for (size_t axis = ndim; axis > 0; --axis) {
    size_t i = axis - 1;
    coords[i] = tmp % out_shape[i];
    tmp /= out_shape[i];
  }

  size_t batch = coords[0];
  size_t out_channel = coords[1];
  size_t out_offset = OffsetFromCoordinates(coords, ndim, out_strides);
  size_t out_channels_per_group = weight_shape[0] / groups;
  size_t input_channels_per_group = weight_shape[1];
  size_t group = out_channel / out_channels_per_group;

  float acc = 0.0f;
  if (has_bias) {
    acc =
        Caster<kDev>::template Cast<float>(bias[out_channel * bias_strides[0]]);
  }

  for (size_t in_group_channel = 0; in_group_channel < input_channels_per_group;
       ++in_group_channel) {
    size_t input_channel = group * input_channels_per_group + in_group_channel;

    for (size_t kernel_linear = 0; kernel_linear < kernel_size;
         ++kernel_linear) {
      size_t kernel_coords[3] = {0, 0, 0};
      size_t rem = kernel_linear;
      bool inside = true;
      size_t input_spatial[3] = {0, 0, 0};

      for (size_t rev = spatial_ndim; rev > 0; --rev) {
        size_t d = rev - 1;
        kernel_coords[d] = rem % weight_shape[d + 2];
        rem /= weight_shape[d + 2];
      }

      for (size_t d = 0; d < spatial_ndim; ++d) {
        int64_t pos = static_cast<int64_t>(coords[d + 2]) * stride[d] -
                      padding[d] +
                      static_cast<int64_t>(kernel_coords[d]) * dilation[d];
        if (pos < 0 || pos >= static_cast<int64_t>(input_shape[d + 2])) {
          inside = false;
          break;
        }
        input_spatial[d] = static_cast<size_t>(pos);
      }

      if (!inside) {
        continue;
      }

      size_t input_coords[5] = {batch, input_channel, 0, 0, 0};
      size_t weight_coords[5] = {out_channel, in_group_channel, 0, 0, 0};
      for (size_t d = 0; d < spatial_ndim; ++d) {
        input_coords[d + 2] = input_spatial[d];
        weight_coords[d + 2] = kernel_coords[d];
      }

      float x = Caster<kDev>::template Cast<float>(
          input[OffsetFromCoordinates(input_coords, ndim, input_strides)]);
      float w = Caster<kDev>::template Cast<float>(
          weight[OffsetFromCoordinates(weight_coords, ndim, weight_strides)]);
      acc += x * w;
    }
  }

  out[out_offset] = Caster<kDev>::template Cast<T>(acc);
}

}  // namespace infini::ops

#endif
