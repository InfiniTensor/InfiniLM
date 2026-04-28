#ifndef __UNFOLD_MOORE_KERNEL_H__
#define __UNFOLD_MOORE_KERNEL_H__

#include <cstdio>
#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <type_traits>

namespace op::unfold::moore {

template <typename T>
__global__ void unfold_kernel(
    T *__restrict__ output,      // [N, C_out, L]
    const T *__restrict__ input, // [N, C, H, W]
    // 维度参数
    int C, int H, int W,  // 输入维度
    int out_h, int out_w, // 输出空间维度
    // 算子参数
    int k_h, int k_w,           // Kernel Size
    int pad_h, int pad_w,       // Padding
    int stride_h, int stride_w, // Stride
    int dil_h, int dil_w,       // Dilation
    // 总任务量
    size_t total_elements) {

    // 平铺式索引：每个线程处理输出的一个元素
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        // --------------------------------------------------------
        // 1. 坐标反算：从线性 idx 解析出逻辑维度
        // 输出形状逻辑上为: [N, (C * kH * kW), (out_h * out_w)]
        // --------------------------------------------------------
        int L = out_h * out_w;
        int kernel_area = k_h * k_w;
        int C_col = C * kernel_area; // 输出的通道数 (Column Channel)

        int l_idx = idx % L;
        size_t temp = idx / L;
        int c_col_idx = temp % C_col;
        int n_idx = temp / C_col;

        // 解析空间坐标 (h_out, w_out)
        int w_out = l_idx % out_w;
        int h_out = l_idx / out_w;

        // 解析通道坐标 -> (c_in, kh, kw)
        int kw = c_col_idx % k_w;
        int temp_k = c_col_idx / k_w;
        int kh = temp_k % k_h;
        int c_in = temp_k / k_h;

        // 计算输入特征图上的坐标
        int h_in = h_out * stride_h - pad_h + kh * dil_h;
        int w_in = w_out * stride_w - pad_w + kw * dil_w;

        T val;

        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
            // 计算输入线性索引：[n, c, h, w]
            size_t in_idx = ((static_cast<size_t>(n_idx) * C + c_in) * H + h_in) * W + w_in;
            val = input[in_idx];
        } else {
            if constexpr (std::is_same_v<T, half>) {
                val = __float2half(0.0f);
            } else if constexpr (std::is_same_v<T, __mt_bfloat16>) {
                val = __float2bfloat16(0.0f);
            } else {
                val = static_cast<T>(0.0f);
            }
        }

        output[idx] = val;
    }
}

} // namespace op::unfold::moore

#endif // __UNFOLD_MOORE_KERNEL_H__
