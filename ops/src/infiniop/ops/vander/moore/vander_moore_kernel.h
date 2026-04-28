#ifndef __VANDER_MOORE_KERNEL_H__
#define __VANDER_MOORE_KERNEL_H__

#include <cmath>
#include <cstdio>
#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <type_traits>

namespace op::vander::moore {

// ==================================================================
// 核心 Kernel
// ==================================================================
template <typename T>
__global__ void vander_kernel(
    T *__restrict__ output,      // [rows, cols]
    const T *__restrict__ input, // [rows]
    size_t rows,
    size_t cols,
    bool increasing) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = rows * cols;

    if (idx < total_elements) {
        size_t row = idx / cols;
        size_t col = idx % cols;

        // 1. 读取输入并转换为 float
        T in_val = input[row];
        float x;

        if constexpr (std::is_same_v<T, half>) {
            x = __half2float(in_val);
        } else if constexpr (std::is_same_v<T, __mt_bfloat16>) {
            x = __bfloat162float(in_val);
        } else {
            x = static_cast<float>(in_val);
        }

        // 2. 计算幂
        float power = increasing ? static_cast<float>(col)
                                 : static_cast<float>(cols - 1 - col);
        float res = powf(x, power);

        // 3. 结果写回
        if constexpr (std::is_same_v<T, half>) {
            output[idx] = __float2half(res);
        } else if constexpr (std::is_same_v<T, __mt_bfloat16>) {
            output[idx] = __float2bfloat16(res);
        } else {
            output[idx] = static_cast<T>(res);
        }
    }
}

} // namespace op::vander::moore

#endif // __VANDER_MOORE_KERNEL_H__
