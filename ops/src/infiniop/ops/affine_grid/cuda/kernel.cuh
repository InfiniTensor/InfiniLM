#ifndef __AFFINE_GRID_CUDA_H__
#define __AFFINE_GRID_CUDA_H__

#include <cmath>
#include <type_traits>

namespace op::affine_grid::cuda {

template <typename T>
__device__ __forceinline__ float to_float_acc(const T &x) {
    if constexpr (std::is_same_v<T, half>) {
        return __half2float(x);
    } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
        return __bfloat162float(x);
    } else {
        return static_cast<float>(x);
    }
}

template <typename T>
__global__ void affine_grid_kernel(
    T *__restrict__ output,      // [优化1] 使用 __restrict__
    const T *__restrict__ theta, // [优化1] 使用 __restrict__
    size_t N,
    size_t H,
    size_t W,
    bool align_corners) {
    // 扁平化索引
    size_t total_elements = N * H * W;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_elements) {
        return;
    }

    float w_scale, h_scale, w_bias, h_bias;

    if (align_corners) {
        // align_corners = True: formula is (2*i)/(size-1) - 1
        // => i * (2/(size-1)) - 1
        w_scale = (W > 1) ? 2.0f / (W - 1.0f) : 0.0f;
        h_scale = (H > 1) ? 2.0f / (H - 1.0f) : 0.0f;
        w_bias = -1.0f;
        h_bias = -1.0f;
    } else {
        // align_corners = False: formula is (2*i + 1)/size - 1
        // => i * (2/size) + (1/size - 1)
        w_scale = 2.0f / W;
        h_scale = 2.0f / H;
        w_bias = 1.0f / W - 1.0f;
        h_bias = 1.0f / H - 1.0f;
    }

    size_t w = idx % W;
    size_t temp = idx / W;
    size_t h = temp % H;
    size_t n = temp / H; // 此时 temp = n * H + h

    // 2. 计算归一化坐标 (使用乘法代替除法)
    float x_norm = (float)w * w_scale + w_bias;
    float y_norm = (float)h * h_scale + h_bias;

    // 如果 align_corners=True 且 size=1，特判修正
    if (align_corners) {
        if (W <= 1) {
            x_norm = 0.0f;
        }
        if (H <= 1) {
            y_norm = 0.0f;
        }
    }

    const T *theta_ptr = theta + n * 6;

    float r00 = to_float_acc(theta_ptr[0]);
    float r01 = to_float_acc(theta_ptr[1]);
    float tx = to_float_acc(theta_ptr[2]);
    float r10 = to_float_acc(theta_ptr[3]);
    float r11 = to_float_acc(theta_ptr[4]);
    float ty = to_float_acc(theta_ptr[5]);

    float grid_x = r00 * x_norm + r01 * y_norm + tx;
    float grid_y = r10 * x_norm + r11 * y_norm + ty;

    // 5. 向量化写入 (Vectorized Store)
    if constexpr (std::is_same_v<T, float>) {
        float2 *out_vec = reinterpret_cast<float2 *>(output);
        out_vec[idx] = make_float2(grid_x, grid_y);
    } else if constexpr (std::is_same_v<T, half>) {
        half2 *out_vec = reinterpret_cast<half2 *>(output);
        out_vec[idx] = __floats2half2_rn(grid_x, grid_y);
    } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
        cuda_bfloat162 *out_vec = reinterpret_cast<cuda_bfloat162 *>(output);
        out_vec[idx] = __floats2bfloat162_rn(grid_x, grid_y);
    } else {
        output[idx * 2 + 0] = static_cast<T>(grid_x);
        output[idx * 2 + 1] = static_cast<T>(grid_y);
    }
}

} // namespace op::affine_grid::cuda

#endif // __AFFINE_GRID_CUDA_H__
