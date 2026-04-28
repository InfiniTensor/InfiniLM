#ifndef __UPSAMPLE_NEAREST_CUDA_CUH__
#define __UPSAMPLE_NEAREST_CUDA_CUH__

#include <cmath>
#include <cstdio>

namespace op::upsample_nearest::cuda {
__device__ __forceinline__ int get_nearest_index(
    int out_index,
    float scale,
    int input_size) {
    int idx = static_cast<int>(floorf(out_index * scale));
    return min(max(idx, 0), input_size - 1);
}
template <typename T>
__global__ void upsample_nearest_kernel(
    T *__restrict__ output,      // [N, C, H_out, W_out]
    const T *__restrict__ input, // [N, C, H_in, W_in]
    size_t N,
    size_t C,
    size_t H_in,
    size_t W_in,
    size_t H_out,
    size_t W_out,
    float scale_h,   // 预计算的缩放比例 (in_size / out_size)
    float scale_w) { // 预计算的缩放比例 (in_size / out_size)
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = N * C * H_out * W_out;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < total_elements; i += stride) {
        // 1. 解构索引 (N, C, H_out, W_out)
        // Layout: NCHW
        size_t w_out_idx = i % W_out;
        size_t temp = i / W_out;
        size_t h_out_idx = temp % H_out;
        temp /= H_out;
        size_t c_idx = temp % C;
        size_t n_idx = temp / C;

        // 2. 计算源索引 (Source Indices)
        int h_in_idx = get_nearest_index(static_cast<int>(h_out_idx), scale_h, static_cast<int>(H_in));
        int w_in_idx = get_nearest_index(static_cast<int>(w_out_idx), scale_w, static_cast<int>(W_in));
        // Input layout: [N, C, H_in, W_in]
        size_t in_offset = (n_idx * C + c_idx) * H_in * W_in + h_in_idx * W_in + w_in_idx;
        output[i] = input[in_offset];
    }
}

} // namespace op::upsample_nearest::cuda

#endif // __UPSAMPLE_NEAREST_CUDA_CUH__
