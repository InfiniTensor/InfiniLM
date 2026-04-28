#ifndef __UPSAMPLE_BILINEAR_CUDA_CUH__
#define __UPSAMPLE_BILINEAR_CUDA_CUH__

#include <cmath>
#include <cstdio>

namespace op::upsample_bilinear::cuda {

// ==================================================================
// 辅助函数: 计算源坐标
// ==================================================================
__device__ __forceinline__ float get_source_coord(
    float scale,
    int out_index,
    bool align_corners) {

    if (align_corners) {
        return static_cast<float>(out_index) * scale;
    } else {
        // formula: (x + 0.5) * scale - 0.5
        return (static_cast<float>(out_index) + 0.5f) * scale - 0.5f;
    }
}

__device__ __forceinline__ int clamp(int val, int min_val, int max_val) {
    return max(min_val, min(val, max_val));
}

// ==================================================================
// Kernel: 双线性插值核心逻辑
// ==================================================================
template <typename T>
__global__ void upsample_bilinear_kernel(
    T *__restrict__ output,      // [N, C, H_out, W_out]
    const T *__restrict__ input, // [N, C, H_in, W_in]
    size_t N,
    size_t C,
    size_t H_in,
    size_t W_in,
    size_t H_out,
    size_t W_out,
    float scale_h, // 预计算的缩放比例
    float scale_w, // 预计算的缩放比例
    bool align_corners) {

    // Grid-Stride Loop: 处理每一个输出元素
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

        // 2. 计算源坐标 (Source Coordinates)
        float h_real = get_source_coord(scale_h, h_out_idx, align_corners);
        float w_real = get_source_coord(scale_w, w_out_idx, align_corners);

        // 3. 计算上下左右四个最近邻整数坐标
        // y (height) direction
        int h0 = static_cast<int>(floorf(h_real));
        int h1 = h0 + 1;
        // x (width) direction
        int w0 = static_cast<int>(floorf(w_real));
        int w1 = w0 + 1;

        // 4. 计算插值权重 (Weights)
        float h1_lambda = h_real - h0;
        float h0_lambda = 1.0f - h1_lambda;
        float w1_lambda = w_real - w0;
        float w0_lambda = 1.0f - w1_lambda;

        // 5. 边界处理 (Clamping)
        h0 = clamp(h0, 0, static_cast<int>(H_in) - 1);
        h1 = clamp(h1, 0, static_cast<int>(H_in) - 1);
        w0 = clamp(w0, 0, static_cast<int>(W_in) - 1);
        w1 = clamp(w1, 0, static_cast<int>(W_in) - 1);

        // 6. 读取数据
        // 计算当前 Batch 和 Channel 的 Input 基地址
        const T *img_base = input + (n_idx * C + c_idx) * H_in * W_in;

        float val00 = static_cast<float>(img_base[h0 * W_in + w0]);
        float val01 = static_cast<float>(img_base[h0 * W_in + w1]);
        float val10 = static_cast<float>(img_base[h1 * W_in + w0]);
        float val11 = static_cast<float>(img_base[h1 * W_in + w1]);

        // 7. 双线性插值计算
        // result = (val00 * w0 + val01 * w1) * h0 + (val10 * w0 + val11 * w1) * h1
        float val = h0_lambda * (w0_lambda * val00 + w1_lambda * val01) + h1_lambda * (w0_lambda * val10 + w1_lambda * val11);

        output[i] = static_cast<T>(val);
    }
}

} // namespace op::upsample_bilinear::cuda

#endif // __UPSAMPLE_BILINEAR_CUDA_CUH__
