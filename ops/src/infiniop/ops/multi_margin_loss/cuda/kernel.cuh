#ifndef __MULTI_MARGIN_LOSS_CUDA_CUH__
#define __MULTI_MARGIN_LOSS_CUDA_CUH__

#include <cmath>
#include <cstdio>

namespace op::multi_margin_loss::cuda {
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
struct MultiMarginLossFunctor {
    int p;
    float margin;

    __host__ __device__ MultiMarginLossFunctor(int p_val, float margin_val)
        : p(p_val), margin(margin_val) {}

    // 计算单个 class c 的 loss 分量
    // diff = margin - target_score + other_score
    __device__ __forceinline__ float compute(float diff) const {
        if (diff > 0.0f) {
            return (p == 1) ? diff : diff * diff;
        }
        return 0.0f;
    }
};
template <typename T>
__global__ void multi_margin_loss_kernel(
    T *__restrict__ output,             // [N]
    const T *__restrict__ input,        // [N, C]
    const int64_t *__restrict__ target, // [N]
    const T *__restrict__ weight,       // [C] (Optional)
    size_t N,
    size_t C,
    MultiMarginLossFunctor functor) {

    size_t n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < N) {
        int64_t target_idx = target[n];

        // 越界检查
        if (target_idx < 0 || target_idx >= static_cast<int64_t>(C)) {
            output[n] = static_cast<T>(0.0f);
            return;
        }

        // 定位当前行的起始位置
        const T *row_ptr = input + n * C;
        float target_score = static_cast<float>(row_ptr[target_idx]);
        float sum_loss = 0.0f;

        // 遍历所有类别
        for (size_t c = 0; c < C; ++c) {
            if (c == static_cast<size_t>(target_idx)) {
                continue;
            }

            float other_score = static_cast<float>(row_ptr[c]);
            float diff = functor.margin - target_score + other_score;
            sum_loss += functor.compute(diff);
        }

        // 公式: sum / C
        sum_loss /= static_cast<float>(C);

        // 应用权重
        if (weight != nullptr) {
            float w = static_cast<float>(weight[target_idx]);
            sum_loss *= w;
        }

        output[n] = static_cast<T>(sum_loss);
    }
}
template <typename T>
__global__ void multi_margin_loss_reduce_kernel(
    float *output,                      // [1] Accumulator (Float)
    const T *__restrict__ input,        // [N, C]
    const int64_t *__restrict__ target, // [N]
    const T *__restrict__ weight,       // [C]
    size_t N,
    size_t C,
    MultiMarginLossFunctor functor,
    float scale // Mean模式传 1/N, Sum模式传 1.0
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    float local_sum = 0.0f;

    // Grid-Stride Loop over Batch Dimension N
    for (size_t n = idx; n < N; n += stride) {
        int64_t target_idx = target[n];

        if (target_idx >= 0 && target_idx < static_cast<int64_t>(C)) {
            const T *row_ptr = input + n * C;
            float target_score = static_cast<float>(row_ptr[target_idx]);
            float sample_loss = 0.0f;

            for (size_t c = 0; c < C; ++c) {
                if (c == static_cast<size_t>(target_idx)) {
                    continue;
                }

                float other_score = static_cast<float>(row_ptr[c]);
                float diff = functor.margin - target_score + other_score;
                sample_loss += functor.compute(diff);
            }

            sample_loss /= static_cast<float>(C);

            if (weight != nullptr) {
                float w = static_cast<float>(weight[target_idx]);
                sample_loss *= w;
            }

            local_sum += sample_loss;
        }
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

} // namespace op::multi_margin_loss::cuda

#endif // __MULTI_MARGIN_LOSS_CUDA_CUH__
