#ifndef __TRIPLET_MARGIN_LOSS_CUDA_CUH__
#define __TRIPLET_MARGIN_LOSS_CUDA_CUH__

#include <cmath>
#include <cstdio>

namespace op::triplet_margin_loss::cuda {

template <typename T, int N>
struct alignas(sizeof(T) * N) Pack {
    T val[N];
};

// ==================================================================
// 归约辅助函数 (Warp & Block Reduction)
// ==================================================================
__device__ __forceinline__ float warpReduceSum(float val) {
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

__device__ __forceinline__ float blockReduceSum(float val) {
    static __shared__ float shared[32]; // Max 1024 threads / 32 warps
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // 假设 BlockDim 也是 32 的倍数
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    if (wid == 0) {
        val = warpReduceSum(val);
    }
    return val;
}

// ==================================================================
// Functor: 核心数学逻辑
// ==================================================================
struct TripletMarginLossFunctor {
    float margin;
    int p;
    float eps;
    bool swap;

    __host__ __device__ TripletMarginLossFunctor(float margin_, int p_, float eps_, bool swap_)
        : margin(margin_), p(p_), eps(eps_), swap(swap_) {}

    // 辅助函数: 计算两个向量 x, y 之间的 p-范数距离
    // x, y 指针，长度 D
    template <typename T>
    __device__ __forceinline__ float compute_dist(const T *x, const T *y, size_t D) const {
        float sum = 0.0f;
        for (size_t i = 0; i < D; ++i) {
            float diff = fabsf(static_cast<float>(x[i]) - static_cast<float>(y[i]));
            if (p == 1) {
                sum += diff;
            } else if (p == 2) {
                sum += diff * diff;
            } else {
                sum += powf(diff, static_cast<float>(p));
            }
        }

        if (p == 1) {
            return sum + eps;
        } else if (p == 2) {
            return sqrtf(sum + eps);
        } else {
            return powf(sum + eps, 1.0f / static_cast<float>(p));
        }
    }

    // 计算单个 Triplet 的 Loss
    __device__ __forceinline__ float compute_loss(float dist_pos, float dist_neg) const {
        float val = dist_pos - dist_neg + margin;
        return (val > 0.0f) ? val : 0.0f; // max(0, val)
    }
};

// ==================================================================
// Kernel 1: Pointwise / No Reduction
// 输出 Tensor 形状 [N]
// ==================================================================
template <typename T>
__global__ void triplet_margin_loss_kernel(
    T *__restrict__ output,         // [N]
    const T *__restrict__ anchor,   // [N, D]
    const T *__restrict__ positive, // [N, D]
    const T *__restrict__ negative, // [N, D]
    size_t N,
    size_t D,
    TripletMarginLossFunctor functor) {

    size_t n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < N) {
        // 定位当前样本的起始位置
        const T *a_ptr = anchor + n * D;
        const T *p_ptr = positive + n * D;
        const T *n_ptr = negative + n * D;

        float dist_pos = functor.compute_dist(a_ptr, p_ptr, D);
        float dist_neg = functor.compute_dist(a_ptr, n_ptr, D);

        // Swap 逻辑: 取 d(p, n) 和 d(a, n) 中较小的作为负样本距离
        if (functor.swap) {
            float dist_swap = functor.compute_dist(p_ptr, n_ptr, D);
            if (dist_swap < dist_neg) {
                dist_neg = dist_swap;
            }
        }

        float loss = functor.compute_loss(dist_pos, dist_neg);
        output[n] = static_cast<T>(loss);
    }
}

// ==================================================================
// Kernel 2: Reduction (Mean / Sum)
// 输出 Scalar (float accumulator -> cast later)
// ==================================================================
template <typename T>
__global__ void triplet_margin_loss_reduce_kernel(
    float *output,                  // [1] Accumulator (Float)
    const T *__restrict__ anchor,   // [N, D]
    const T *__restrict__ positive, // [N, D]
    const T *__restrict__ negative, // [N, D]
    size_t N,
    size_t D,
    TripletMarginLossFunctor functor,
    float scale // Mean模式传 1/N, Sum模式传 1.0
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    float local_sum = 0.0f;

    // Grid-Stride Loop over Batch Dimension N
    for (size_t n = idx; n < N; n += stride) {
        const T *a_ptr = anchor + n * D;
        const T *p_ptr = positive + n * D;
        const T *n_ptr = negative + n * D;

        float dist_pos = functor.compute_dist(a_ptr, p_ptr, D);
        float dist_neg = functor.compute_dist(a_ptr, n_ptr, D);

        if (functor.swap) {
            float dist_swap = functor.compute_dist(p_ptr, n_ptr, D);
            if (dist_swap < dist_neg) {
                dist_neg = dist_swap;
            }
        }

        local_sum += functor.compute_loss(dist_pos, dist_neg);
    }

    // Block Reduction
    float block_sum = blockReduceSum(local_sum);

    // Global Atomic Add (Reduce to scalar)
    if (threadIdx.x == 0) {
        atomicAdd(output, block_sum * scale);
    }
}
template <typename T>
__global__ void cast_float_to_t(T *output, const float *src) {
    *output = static_cast<T>(*src);
}

} // namespace op::triplet_margin_loss::cuda

#endif // __TRIPLET_MARGIN_LOSS_CUDA_CUH__
