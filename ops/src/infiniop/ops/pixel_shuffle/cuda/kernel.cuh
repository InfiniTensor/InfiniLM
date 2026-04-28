#pragma once
#include <cstddef>
#include <type_traits>

namespace op::cuda {

template <typename T>
__global__ void pixel_shuffle_kernel(
    T *output,
    const T *input,
    size_t batch,
    size_t out_channels,
    size_t height,
    size_t width,
    int r) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch * out_channels * height * width;

    if (idx >= total) {
        return;
    }

    size_t n = idx / (out_channels * height * width);
    size_t rem = idx % (out_channels * height * width);
    size_t c = rem / (height * width);
    rem = rem % (height * width);
    size_t oh = rem / width;
    size_t ow = rem % width;

    // Calculate input indices
    size_t w = ow / r;
    size_t h = oh / r;
    size_t i = oh % r;
    size_t j = ow % r;
    size_t in_c = c * r * r + i * r + j;

    size_t in_idx = ((n * (out_channels * r * r) + in_c) * (height / r) + h) * (width / r) + w;
    size_t out_idx = ((n * out_channels + c) * height + oh) * width + ow;

    output[out_idx] = input[in_idx];
}

template <typename T>
__global__ void pixel_shuffle_kernel_strided(
    T *output,
    const T *input,
    size_t batch,
    size_t out_channels,
    size_t out_height,
    size_t out_width,
    int r,
    ptrdiff_t x_stride0,
    ptrdiff_t x_stride1,
    ptrdiff_t x_stride2,
    ptrdiff_t x_stride3,
    ptrdiff_t y_stride0,
    ptrdiff_t y_stride1,
    ptrdiff_t y_stride2,
    ptrdiff_t y_stride3) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch * out_channels * out_height * out_width;

    if (idx >= total) {
        return;
    }

    const size_t spatial = out_height * out_width;
    const size_t chw = out_channels * spatial;

    size_t n = idx / chw;
    size_t rem = idx % chw;
    size_t c = rem / spatial;
    rem = rem % spatial;
    size_t oh = rem / out_width;
    size_t ow = rem % out_width;

    const size_t upscale = static_cast<size_t>(r);
    const size_t ih = oh / upscale;
    const size_t iw = ow / upscale;
    const size_t i = oh % upscale;
    const size_t j = ow % upscale;
    const size_t in_c = c * upscale * upscale + i * upscale + j;

    const ptrdiff_t in_offset = static_cast<ptrdiff_t>(n) * x_stride0 + static_cast<ptrdiff_t>(in_c) * x_stride1 + static_cast<ptrdiff_t>(ih) * x_stride2 + static_cast<ptrdiff_t>(iw) * x_stride3;
    const ptrdiff_t out_offset = static_cast<ptrdiff_t>(n) * y_stride0 + static_cast<ptrdiff_t>(c) * y_stride1 + static_cast<ptrdiff_t>(oh) * y_stride2 + static_cast<ptrdiff_t>(ow) * y_stride3;

    output[out_offset] = input[in_offset];
}

} // namespace op::cuda
