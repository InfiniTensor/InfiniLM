#ifndef __ADAPTIVE_MAX_POOL1D_CUDA_KERNEL_H__
#define __ADAPTIVE_MAX_POOL1D_CUDA_KERNEL_H__

#include <cmath>
#include <limits>

template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
__device__ void adaptiveMaxPool1dBlock(
    Tdata *__restrict__ y,
    ptrdiff_t stride_y_batch,
    ptrdiff_t stride_y_channel,
    const Tdata *__restrict__ x,
    ptrdiff_t stride_x_batch,
    ptrdiff_t stride_x_channel,
    ptrdiff_t stride_x_length,
    size_t channels,
    size_t input_length,
    size_t output_length,
    size_t ndim) {

    size_t block_idx = blockIdx.x;
    size_t batch_idx = block_idx / channels;
    size_t channel_idx = block_idx % channels;

    const Tdata *x_ptr;
    Tdata *y_ptr;

    if (ndim > 2) {
        x_ptr = x + batch_idx * stride_x_batch + channel_idx * stride_x_channel;
        y_ptr = y + batch_idx * stride_y_batch + channel_idx * stride_y_channel;
    } else {
        x_ptr = x + batch_idx * stride_x_batch;
        y_ptr = y + batch_idx * stride_y_batch;
    }

    for (size_t out_idx = threadIdx.x; out_idx < output_length; out_idx += BLOCK_SIZE) {
        int start_index = static_cast<int>(floorf((float)out_idx * input_length / output_length));
        int end_index = static_cast<int>(ceilf((float)(out_idx + 1) * input_length / output_length));

        if (end_index <= start_index) {
            continue;
        }

        Tcompute max_val = Tcompute(x_ptr[start_index * stride_x_length]);
        for (int i = start_index + 1; i < end_index; ++i) {
            Tcompute val = Tcompute(x_ptr[i * stride_x_length]);
            max_val = max(max_val, val);
        }

        y_ptr[out_idx] = Tdata(max_val);
    }
}

#endif
