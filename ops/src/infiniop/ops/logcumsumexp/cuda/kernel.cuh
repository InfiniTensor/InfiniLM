#ifndef __LOGCUMSUMEXP_CUDA_CUH__
#define __LOGCUMSUMEXP_CUDA_CUH__

#include <cmath>
#include <limits>

namespace op::logcumsumexp::cuda {

// ============================================================
// 数值稳定 LogSumExp prefix state
// 等价于：log(sum(exp(x[0:i])))
// ============================================================

struct LSEState {
    float m; // running max
    float s; // sum(exp(x - m))

    // 数学单位元：log(0) = -inf
    __device__ __forceinline__ static LSEState identity() {
        return {-INFINITY, 0.0f};
    }

    // prefix 更新
    __device__ __forceinline__ void update(float v) {
        if (m == -INFINITY) {
            // 第一个元素
            m = v;
            s = 1.0f;
        } else if (v > m) {
            // max 发生变化，需要 rescale
            s = s * expf(m - v) + 1.0f;
            m = v;
        } else {
            s += expf(v - m);
        }
    }

    // 当前 log-sum-exp 值
    __device__ __forceinline__ float value() const {
        return (s == 0.0f) ? -INFINITY : (m + logf(s));
    }
};

// ============================================================
// kernel：一个 thread 负责一个 (outer, inner) 前缀向量
// ============================================================

template <typename T>
__global__ void logcumsumexp_kernel(
    T *__restrict__ y,
    const T *__restrict__ x,

    size_t outer_size,
    size_t axis_size,
    size_t inner_size,

    size_t x_axis_stride,
    size_t x_inner_stride,
    size_t x_outer_stride,

    size_t y_axis_stride,
    size_t y_inner_stride,
    size_t y_outer_stride,

    bool exclusive,
    bool reverse) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t num_vec = outer_size * inner_size;
    if (tid >= num_vec) {
        return;
    }

    size_t o = tid / inner_size;
    size_t i = tid % inner_size;

    // base offset（正确处理 stride）
    size_t x_base = o * x_outer_stride + i * x_inner_stride;
    size_t y_base = o * y_outer_stride + i * y_inner_stride;

    LSEState state = LSEState::identity();

    for (size_t k = 0; k < axis_size; ++k) {
        size_t kk = reverse ? (axis_size - 1 - k) : k;

        size_t x_off = x_base + kk * x_axis_stride;
        size_t y_off = y_base + kk * y_axis_stride;

        float v = static_cast<float>(x[x_off]);

        if (exclusive) {
            // y[i] = log(sum(exp(x[:i])))
            y[y_off] = static_cast<T>(state.value());
            state.update(v);
        } else {
            // y[i] = log(sum(exp(x[:i+1])))
            state.update(v);
            y[y_off] = static_cast<T>(state.value());
        }

        // ===== 调试用（需要时打开）=====
        /*
        if (o == 0 && i == 0 && k < 5) {
            printf(
                "[CUDA] k=%zu v=%f m=%f s=%f out=%f\n",
                k, v, state.m, state.s,
                static_cast<float>(y[y_off])
            );
        }
        */
    }
}

} // namespace op::logcumsumexp::cuda

#endif // __LOGCUMSUMEXP_CUDA_CUH__
