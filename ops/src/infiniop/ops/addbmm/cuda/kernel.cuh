#ifndef __ADDBMM_NVIDIA_CUH__
#define __ADDBMM_NVIDIA_CUH__

#include <type_traits>

namespace op::addbmm::nvidia {

// --- 常量定义 ---
constexpr int BLOCK_SIZE = 16; // 16x16 线程块，处理 FP32/FP16 比较通用

template <typename T>
__device__ __forceinline__ float to_float_acc(const T &x) {
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
__device__ __forceinline__ T from_float_res(float x) {
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

template <typename T>
__global__ void addbmm_tiled_kernel(
    T *output,
    const T *input,
    const T *batch1,
    const T *batch2,
    size_t b, size_t n, size_t m, size_t p,
    float alpha, float beta,
    // Strides
    ptrdiff_t out_s0, ptrdiff_t out_s1,
    ptrdiff_t inp_s0, ptrdiff_t inp_s1,
    ptrdiff_t b1_s0, ptrdiff_t b1_s1, ptrdiff_t b1_s2,
    ptrdiff_t b2_s0, ptrdiff_t b2_s1, ptrdiff_t b2_s2) {
    // Block 行列索引 (Output 的坐标)
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float acc = 0.0f;

    // Shared Memory 缓存
    // 大小: 2个矩阵 * BLOCK_SIZE * BLOCK_SIZE
    __shared__ float s_b1[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_b2[BLOCK_SIZE][BLOCK_SIZE];

    // 遍历每一个 Batch
    for (int batch_idx = 0; batch_idx < b; ++batch_idx) {

        // 遍历 K 维度 (即 m 维度)，步长为 BLOCK_SIZE
        for (int k = 0; k < m; k += BLOCK_SIZE) {

            if (row < n && (k + threadIdx.x) < m) {
                // b1: [batch_idx, row, k + tx]
                size_t idx = batch_idx * b1_s0 + row * b1_s1 + (k + threadIdx.x) * b1_s2;
                s_b1[threadIdx.y][threadIdx.x] = to_float_acc(batch1[idx]);
            } else {
                s_b1[threadIdx.y][threadIdx.x] = 0.0f;
            }

            if ((k + threadIdx.y) < m && col < p) {
                // b2: [batch_idx, k + ty, col]
                size_t idx = batch_idx * b2_s0 + (k + threadIdx.y) * b2_s1 + col * b2_s2;
                s_b2[threadIdx.y][threadIdx.x] = to_float_acc(batch2[idx]);
            } else {
                s_b2[threadIdx.y][threadIdx.x] = 0.0f;
            }

            // 等待所有线程加载完毕
            __syncthreads();

// 2. 计算子块乘积 (Partial Accumulation)
// 循环展开以提高指令吞吐量
#pragma unroll
            for (int e = 0; e < BLOCK_SIZE; ++e) {
                acc += s_b1[threadIdx.y][e] * s_b2[e][threadIdx.x];
            }

            // 等待计算完毕，准备加载下一个 tile
            __syncthreads();
        }
    }

    // 3. 写入结果 (Scale + Add Input)
    if (row < n && col < p) {
        float res = 0.0f;

        // Input 部分: beta * input
        if (beta != 0.0f) {
            size_t inp_idx = row * inp_s0 + col * inp_s1;
            res = beta * to_float_acc(input[inp_idx]);
        }

        // Matmul 部分: alpha * sum
        res += alpha * acc;

        size_t out_idx = row * out_s0 + col * out_s1;
        output[out_idx] = from_float_res<T>(res);
    }
}

// ==================================================================
// 2. Launcher
// ==================================================================
template <typename T>
void launch_kernel(
    void *output, const void *input, const void *batch1, const void *batch2,
    size_t b, size_t n, size_t m, size_t p,
    float alpha, float beta,
    // Strides
    ptrdiff_t out_s0, ptrdiff_t out_s1,
    ptrdiff_t inp_s0, ptrdiff_t inp_s1,
    ptrdiff_t b1_s0, ptrdiff_t b1_s1, ptrdiff_t b1_s2,
    ptrdiff_t b2_s0, ptrdiff_t b2_s1, ptrdiff_t b2_s2,
    void *stream) {

    auto out_ptr = reinterpret_cast<T *>(output);
    auto in_ptr = reinterpret_cast<const T *>(input);
    auto b1_ptr = reinterpret_cast<const T *>(batch1);
    auto b2_ptr = reinterpret_cast<const T *>(batch2);

    // 2D Grid 配置
    dim3 block(BLOCK_SIZE, BLOCK_SIZE); // 16x16 threads
    dim3 grid(
        (p + BLOCK_SIZE - 1) / BLOCK_SIZE, // x轴覆盖 col (p)
        (n + BLOCK_SIZE - 1) / BLOCK_SIZE  // y轴覆盖 row (n)
    );

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    addbmm_tiled_kernel<T><<<grid, block, 0, cuda_stream>>>(
        out_ptr, in_ptr, b1_ptr, b2_ptr,
        b, n, m, p,
        alpha, beta,
        out_s0, out_s1, inp_s0, inp_s1,
        b1_s0, b1_s1, b1_s2, b2_s0, b2_s1, b2_s2);
}

// // tiled version fails in some cases
template <typename T>
__global__ void addbmm_kernel(
    const size_t B, const size_t N, const size_t M, const size_t P,
    const float alpha, const float beta,
    T *output,
    const T *input,
    const T *batch1,
    const T *batch2,
    const int64_t out_s0, const int64_t out_s1,
    const int64_t in_s0, const int64_t in_s1,
    const int64_t b1_s0, const int64_t b1_s1, const int64_t b1_s2,
    const int64_t b2_s0, const int64_t b2_s1, const int64_t b2_s2) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = N * P;

    if (idx < total_elements) {
        size_t n = idx / P;
        size_t p = idx % P;

        float matmul_sum = 0.0f;

        int64_t b1_n_offset = n * b1_s1;
        int64_t b2_p_offset = p * b2_s2;

        for (size_t b = 0; b < B; ++b) {
            int64_t b1_b_offset = b * b1_s0;
            int64_t b2_b_offset = b * b2_s0;

            for (size_t m = 0; m < M; ++m) {
                // 直接计算偏移：Batch1[b, n, m]
                int64_t offset1 = b1_b_offset + b1_n_offset + m * b1_s2;
                // 直接计算偏移：Batch2[b, m, p]
                int64_t offset2 = b2_b_offset + m * b2_s1 + b2_p_offset;

                T val1 = batch1[offset1];
                T val2 = batch2[offset2];

                float v1_f, v2_f;
                if constexpr (std::is_same_v<T, half>) {
                    v1_f = __half2float(val1);
                    v2_f = __half2float(val2);
                } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
                    v1_f = __bfloat162float(val1);
                    v2_f = __bfloat162float(val2);
                } else {
                    v1_f = static_cast<float>(val1);
                    v2_f = static_cast<float>(val2);
                }
                matmul_sum += v1_f * v2_f;
            }
        }

        // 直接计算偏移：Input[n, p]
        int64_t in_offset = n * in_s0 + p * in_s1;
        T in_val = input[in_offset];

        float in_val_f;
        if constexpr (std::is_same_v<T, half>) {
            in_val_f = __half2float(in_val);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            in_val_f = __bfloat162float(in_val);
        } else {
            in_val_f = static_cast<float>(in_val);
        }

        float result = beta * in_val_f + alpha * matmul_sum;

        // 直接计算偏移：Output[n, p]
        int64_t out_offset = n * out_s0 + p * out_s1;

        if constexpr (std::is_same_v<T, half>) {
            output[out_offset] = __float2half(result);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            output[out_offset] = __float2bfloat16(result);
        } else {
            output[out_offset] = static_cast<T>(result);
        }
    }
}

// ==================================================================
// 2. Launcher Implementation
// ==================================================================

template <typename T>
void launch_addbmm_naive(
    void *output, const void *input,
    const void *batch1, const void *batch2,
    size_t b, size_t n, size_t m, size_t p,
    float alpha, float beta,
    ptrdiff_t out_s0, ptrdiff_t out_s1,
    ptrdiff_t inp_s0, ptrdiff_t inp_s1,
    ptrdiff_t b1_s0, ptrdiff_t b1_s1, ptrdiff_t b1_s2,
    ptrdiff_t b2_s0, ptrdiff_t b2_s1, ptrdiff_t b2_s2,
    void *stream) {

    size_t total_elements = n * p;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    addbmm_kernel<T><<<blocks, threads, 0, (cudaStream_t)stream>>>(
        b, n, m, p,
        alpha, beta,
        (T *)output,
        (const T *)input,
        (const T *)batch1,
        (const T *)batch2,
        out_s0, out_s1,
        inp_s0, inp_s1,
        b1_s0, b1_s1, b1_s2,
        b2_s0, b2_s1, b2_s2);
}

} // namespace op::addbmm::nvidia

#endif // __ADDBMM_NVIDIA_CUH__
