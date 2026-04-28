#ifndef __VANDER_CUDA_CUH__
#define __VANDER_CUDA_CUH__

#include <cmath>
#include <cstdio>

namespace op::vander::cuda {

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

        // 加载输入 (同一个 row 的不同 col 线程会读取同一个 input[row]，L1 Cache 友好)
        float x = static_cast<float>(input[row]);
        float power = increasing ? static_cast<float>(col)
                                 : static_cast<float>(cols - 1 - col);
        float res = powf(x, power);

        output[idx] = static_cast<T>(res);
    }
}

} // namespace op::vander::cuda

#endif // __VANDER_CUDA_CUH__
