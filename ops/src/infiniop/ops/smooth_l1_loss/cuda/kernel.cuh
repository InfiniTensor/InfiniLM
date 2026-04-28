#ifndef __SMOOTH_L1_LOSS_CUDA_CUH__
#define __SMOOTH_L1_LOSS_CUDA_CUH__

#include <cmath>

namespace op::smooth_l1_loss::cuda {

// ==================================================================
// 基础定义
// ==================================================================
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

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    if (wid == 0) {
        val = warpReduceSum(val);
    }
    return val;
}

// ==================================================================
// Functor
// ==================================================================
struct SmoothL1LossFunctor {
    float beta;
    float inv_beta;
    float half_beta;

    __host__ __device__ SmoothL1LossFunctor(float beta_val)
        : beta(beta_val), inv_beta(1.0f / beta_val), half_beta(0.5f * beta_val) {}

    template <typename T>
    __device__ __forceinline__ float compute(const T &input, const T &target) const {
        float in_f = static_cast<float>(input);
        float tg_f = static_cast<float>(target);
        float diff = in_f - tg_f;
        float abs_diff = fabsf(diff);

        if (abs_diff < beta) {
            return 0.5f * diff * diff * inv_beta;
        } else {
            return abs_diff - half_beta;
        }
    }
};

// ==================================================================
// 1. Elementwise Kernels (reduction='none')
// ==================================================================
template <typename T>
__global__ void smooth_l1_loss_kernel(
    T *__restrict__ output, const T *__restrict__ input, const T *__restrict__ target,
    size_t numel, SmoothL1LossFunctor functor) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = static_cast<T>(functor.compute(input[idx], target[idx]));
    }
}

template <typename T, int PackSize>
__global__ void smooth_l1_loss_kernel_vectorized(
    T *__restrict__ output, const T *__restrict__ input, const T *__restrict__ target,
    size_t num_packs, SmoothL1LossFunctor functor) {
    using PackType = Pack<T, PackSize>;
    auto out_vec = reinterpret_cast<PackType *>(output);
    auto in_vec = reinterpret_cast<const PackType *>(input);
    auto tar_vec = reinterpret_cast<const PackType *>(target);
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_packs) {
        PackType in_pack = in_vec[idx];
        PackType tar_pack = tar_vec[idx];
        PackType out_pack;
#pragma unroll
        for (int i = 0; i < PackSize; ++i) {
            out_pack.val[i] = static_cast<T>(functor.compute(in_pack.val[i], tar_pack.val[i]));
        }
        out_vec[idx] = out_pack;
    }
}

// ==================================================================
// 2. Reduction Kernel (reduction='mean' / 'sum')
// ==================================================================
// 简单的 AtomicAdd 全局归约
template <typename T>
__global__ void smooth_l1_loss_reduce_kernel(
    float *output, // 使用 float 累加防止溢出
    const T *__restrict__ input,
    const T *__restrict__ target,
    size_t numel,
    SmoothL1LossFunctor functor,
    float scale // Mean模式传 1/N, Sum模式传 1.0
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    float local_sum = 0.0f;

    // Grid-Stride Loop
    for (size_t i = idx; i < numel; i += stride) {
        local_sum += functor.compute(input[i], target[i]);
    }

    // Block Reduction
    float block_sum = blockReduceSum(local_sum);

    // Global Atomic Add
    if (threadIdx.x == 0) {
        atomicAdd(output, block_sum * scale);
    }
}

// 辅助 Kernel: 将 float 结果转回目标类型 T 并写入 output
template <typename T>
__global__ void cast_float_to_t(T *output, const float *src) {
    *output = static_cast<T>(*src);
}

} // namespace op::smooth_l1_loss::cuda

#endif // __SMOOTH_L1_LOSS_CUDA_CUH__
