#ifndef __AFFINE_GRID_MOORE_KERNEL_H__
#define __AFFINE_GRID_MOORE_KERNEL_H__

#include <musa_bf16.h> // 包含 __mt_bfloat16 定义
#include <musa_fp16.h>
#include <musa_runtime.h>

namespace op::affine_grid::moore {
typedef struct AffineGridOp {
public:
    static constexpr size_t num_dimensions = 2;

    template <typename T>
    __device__ __forceinline__ void operator()(
        const int w_idx, const int h_idx,
        const int W, const int H,
        const T *theta,
        const bool align_corners,
        T *out_x, T *out_y) const {
        // 1. 归一化坐标计算
        float x_norm, y_norm;
        if (align_corners) {
            x_norm = (float)(w_idx * 2 - (W - 1)) / (float)(MAX(W - 1, 1));
            y_norm = (float)(h_idx * 2 - (H - 1)) / (float)(MAX(H - 1, 1));
        } else {
            x_norm = (float)(w_idx * 2 + 1) / (float)W - 1.0f;
            y_norm = (float)(h_idx * 2 + 1) / (float)H - 1.0f;
        }

        // 2. 仿射变换逻辑
        if constexpr (std::is_same_v<T, half>) {
            float t00 = __half2float(theta[0]);
            float t01 = __half2float(theta[1]);
            float t02 = __half2float(theta[2]);
            float t10 = __half2float(theta[3]);
            float t11 = __half2float(theta[4]);
            float t12 = __half2float(theta[5]);

            float res_x = t00 * x_norm + t01 * y_norm + t02;
            float res_y = t10 * x_norm + t11 * y_norm + t12;

            *out_x = __float2half(res_x);
            *out_y = __float2half(res_y);

        } else if constexpr (std::is_same_v<T, __mt_bfloat16>) { // 【修改】使用 __mt_bfloat16
            // 显式转换 __mt_bfloat16 -> float
            float t00 = __bfloat162float(theta[0]);
            float t01 = __bfloat162float(theta[1]);
            float t02 = __bfloat162float(theta[2]);
            float t10 = __bfloat162float(theta[3]);
            float t11 = __bfloat162float(theta[4]);
            float t12 = __bfloat162float(theta[5]);

            float res_x = t00 * x_norm + t01 * y_norm + t02;
            float res_y = t10 * x_norm + t11 * y_norm + t12;

            // 转换回 __mt_bfloat16
            *out_x = __float2bfloat16(res_x);
            *out_y = __float2bfloat16(res_y);

        } else if constexpr (std::is_same_v<T, float>) {
            float res_x = __fadd_rn(__fmul_rn(theta[0], x_norm), __fadd_rn(__fmul_rn(theta[1], y_norm), theta[2]));
            float res_y = __fadd_rn(__fmul_rn(theta[3], x_norm), __fadd_rn(__fmul_rn(theta[4], y_norm), theta[5]));
            *out_x = res_x;
            *out_y = res_y;
        } else {
            *out_x = theta[0] * static_cast<T>(x_norm) + theta[1] * static_cast<T>(y_norm) + theta[2];
            *out_y = theta[3] * static_cast<T>(x_norm) + theta[4] * static_cast<T>(y_norm) + theta[5];
        }
    }

private:
    __device__ __forceinline__ int MAX(int a, int b) const {
        return a > b ? a : b;
    }

} AffineGridOp;
} // namespace op::affine_grid::moore

#endif // __AFFINE_GRID_MOORE_KERNEL_H__
