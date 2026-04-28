#ifndef __ADAPTIVE_AVG_POOL1D_CUDA_H__
#define __ADAPTIVE_AVG_POOL1D_CUDA_H__

#include <cmath>
#include <type_traits>

namespace op::adaptive_avg_pool1d::cuda {

// -------------------------------------------
// 工具：Warp 级归约求和
// -------------------------------------------
template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// -------------------------------------------
// 工具：数值转换
// -------------------------------------------
template <typename T>
__device__ __forceinline__ float to_float(const T &x) {
    if constexpr (std::is_same_v<T, half>) {
        return __half2float(x);
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
        return __bfloat162float(x);
    }
#endif
    else {
        return static_cast<float>(x);
    }
}

template <typename T>
__device__ __forceinline__ T from_float(float x) {
    if constexpr (std::is_same_v<T, half>) {
        return __float2half(x);
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
        return __float2bfloat16(x);
    }
#endif
    else {
        return static_cast<T>(x);
    }
}

// -------------------------------------------
// Optimization 1: Global Average Pool 特化 Kernel (osize == 1)
// 策略：1 Block 处理 1 个 Channel，Block 内归约
// -------------------------------------------
template <typename T>
__global__ void global_avg_pool1d_kernel(
    T *output,
    const T *input,
    size_t total_channels, // batch * channels
    size_t isize) {
    // 每一个 Block 处理一个 (Batch, Channel) 任务
    size_t channel_idx = blockIdx.x;
    if (channel_idx >= total_channels) {
        return;
    }

    const T *channel_input = input + channel_idx * isize;
    float sum = 0.0f;

    // Grid-Stride Loop within the channel (handle isize > blockDim.x)
    for (size_t i = threadIdx.x; i < isize; i += blockDim.x) {
        sum += to_float(channel_input[i]);
    }

    // Block 内归约
    // 1. Warp Reduce
    sum = warp_reduce_sum(sum);

    // 2. Shared Memory Reduce (跨 Warp)
    static __shared__ float shared_sum[32]; // Max 1024 threads / 32 = 32 warps
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    if (lane == 0) {
        shared_sum[wid] = sum;
    }
    __syncthreads();

    // 3. 让第一个 Warp 把 shared memory 里的结果加起来
    if (wid == 0) {
        float val = (threadIdx.x < (blockDim.x + 31) / 32) ? shared_sum[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (threadIdx.x == 0) {
            output[channel_idx] = from_float<T>(val / static_cast<float>(isize));
        }
    }
}

// -------------------------------------------
// Optimization 2: 通用 Kernel
// 策略：使用浮点数计算索引，避免整数除法；移除不安全的向量化
// -------------------------------------------
template <typename T>
__global__ void adaptive_avg_pool1d_general_kernel(
    T *output,
    const T *input,
    size_t batch_channels,
    size_t isize,
    size_t osize) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    size_t total_elements = batch_channels * osize;

    // 预计算缩放因子，避免循环内除法
    float stride_factor = static_cast<float>(isize) / static_cast<float>(osize);

    for (; idx < total_elements; idx += stride) {
        size_t bc_idx = idx / osize;
        size_t out_idx = idx % osize;

        const T *in_ptr = input + bc_idx * isize;

        // 使用浮点数计算起止点，替代 (i * isize) / osize
        // 注意：PyTorch 官方实现使用 float 进行索引计算
        int istart = static_cast<int>(floorf(out_idx * stride_factor));
        int iend = static_cast<int>(ceilf((out_idx + 1) * stride_factor));

        // 边界保护
        istart = max(0, istart);
        iend = min(static_cast<int>(isize), iend);

        float sum = 0.0f;
        int klen = iend - istart;

        // 标量循环：利用 L1 Cache 和编译器优化
        // 移除手动向量化，因为 istart 不保证对齐
        for (int i = istart; i < iend; ++i) {
            sum += to_float(in_ptr[i]);
        }

        output[idx] = (klen > 0) ? from_float<T>(sum / klen) : from_float<T>(0.0f);
    }
}

// -------------------------------------------
// Launcher
// -------------------------------------------
template <typename T>
void launch_adaptive_avg_pool1d(
    T *output,
    const T *input,
    size_t batch_channels,
    size_t isize,
    size_t osize,
    cudaStream_t stream) {
    // 策略分发
    if (osize == 1) {
        // Case 1: Global Average Pooling (Gap)
        // 每个 Block 处理一个 Channel
        int threads = 256;
        // 如果 isize 很小，减少线程数
        if (isize < 256) {
            threads = 128;
        }
        if (isize < 128) {
            threads = 64;
        }
        if (isize < 64) {
            threads = 32;
        }

        dim3 block(threads);
        dim3 grid(batch_channels);

        global_avg_pool1d_kernel<T><<<grid, block, 0, stream>>>(
            output, input, batch_channels, isize);
    } else {
        // Case 2: General Case
        // 这里的并行度基于输出元素个数
        size_t total_output = batch_channels * osize;
        int threads = 256;
        int blocks = (total_output + threads - 1) / threads;

        // 限制最大 Grid 大小，防止超限
        if (blocks > 65535) {
            blocks = 65535;
        }

        adaptive_avg_pool1d_general_kernel<T><<<blocks, threads, 0, stream>>>(
            output, input, batch_channels, isize, osize);
    }
}

} // namespace op::adaptive_avg_pool1d::cuda

#endif // __ADAPTIVE_AVG_POOL1D_CUDA_H__
