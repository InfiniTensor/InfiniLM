#ifndef __INFINIOP_AVG_POOL1D_CUDA_KERNEL_CUH__
#define __INFINIOP_AVG_POOL1D_CUDA_KERNEL_CUH__

template <typename T>
__device__ void avgPool1dKernel(
    T *y,
    const T *x,
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

    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_elements;
         idx += gridDim.x * blockDim.x) {

        size_t ow = idx % out_width;
        size_t temp = idx / out_width;
        size_t c = temp % channels;
        size_t b = temp / channels;

        size_t y_offset = b * y_stride_batch + c * y_stride_channel + ow * y_stride_width;

        long long start_w = static_cast<long long>(ow * stride) - padding;

        T sum = 0;

        for (size_t k = 0; k < kernel_size; ++k) {
            long long iw = start_w + k;

            if (iw >= 0 && iw < static_cast<long long>(in_width)) {
                size_t x_offset = b * x_stride_batch + c * x_stride_channel + iw * x_stride_width;
                sum += x[x_offset];
            }
        }

#if defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API) || defined(ENABLE_QY_API)
        // Iluvatar __half doesn't accept size_t directly.
        y[y_offset] = sum / static_cast<T>(static_cast<double>(kernel_size));
#else
        y[y_offset] = sum / static_cast<T>(kernel_size);
#endif
    }
}

#endif
