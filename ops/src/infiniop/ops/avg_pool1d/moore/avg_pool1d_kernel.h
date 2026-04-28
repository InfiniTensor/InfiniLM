#ifndef __INFINIOP_AVG_POOL1D_MOORE_KERNEL_H__
#define __INFINIOP_AVG_POOL1D_MOORE_KERNEL_H__

#include <type_traits>

namespace op::avg_pool1d::moore {

template <typename Tdata, typename Tcompute>
__device__ __forceinline__ Tdata castToOutput(Tcompute val) {
    if constexpr (std::is_same_v<Tdata, half>) {
        return __float2half(static_cast<float>(val));
    } else if constexpr (std::is_same_v<Tdata, cuda_bfloat16>) {
        return __float2bfloat16_rn(static_cast<float>(val));
    } else {
        return static_cast<Tdata>(val);
    }
}

template <typename Tdata, typename Tcompute>
__device__ void avgPool1dKernel(
    Tdata *y,
    const Tdata *x,
    size_t batch,
    size_t channels,
    size_t in_width,
    size_t out_width,
    size_t kernel_size,
    size_t stride,
    size_t padding,
    ptrdiff_t y_stride_batch,
    ptrdiff_t y_stride_channel,
    ptrdiff_t y_stride_width,
    ptrdiff_t x_stride_batch,
    ptrdiff_t x_stride_channel,
    ptrdiff_t x_stride_width) {

    size_t total_elements = batch * channels * out_width;
    Tcompute inv_kernel = Tcompute(1) / static_cast<Tcompute>(kernel_size);

    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_elements;
         idx += gridDim.x * blockDim.x) {

        size_t ow = idx % out_width;
        size_t temp = idx / out_width;
        size_t c = temp % channels;
        size_t b = temp / channels;

        size_t y_offset = b * y_stride_batch + c * y_stride_channel + ow * y_stride_width;
        size_t x_base = b * x_stride_batch + c * x_stride_channel;

        long long start_w = static_cast<long long>(ow * stride) - static_cast<long long>(padding);
        long long end_w = start_w + static_cast<long long>(kernel_size);
        long long iw_start = start_w < 0 ? 0 : start_w;
        long long iw_end = end_w > static_cast<long long>(in_width) ? static_cast<long long>(in_width) : end_w;

        Tcompute sum = Tcompute(0);
        if (iw_start < iw_end) {
            size_t x_offset = x_base + static_cast<size_t>(iw_start) * x_stride_width;
            for (long long iw = iw_start; iw < iw_end; ++iw) {
                sum += static_cast<Tcompute>(x[x_offset]);
                x_offset += x_stride_width;
            }
        }

        y[y_offset] = castToOutput<Tdata, Tcompute>(sum * inv_kernel);
    }
}

} // namespace op::avg_pool1d::moore

#endif // __INFINIOP_AVG_POOL1D_MOORE_KERNEL_H__
