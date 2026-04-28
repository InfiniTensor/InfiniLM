#ifndef __SOFTSIGN_MOORE_KERNEL_H__
#define __SOFTSIGN_MOORE_KERNEL_H__

#include <cmath>
#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <type_traits>

namespace op::softsign::moore {

// ================================================================
// 类型转换辅助函数
// ================================================================
template <typename T>
__device__ __forceinline__ float to_float(T v) {
    if constexpr (std::is_same_v<T, half>) {
        return __half2float(v);
    } else if constexpr (std::is_same_v<T, __mt_bfloat16>) {
        return __bfloat162float(v);
    } else {
        return static_cast<float>(v);
    }
}

template <typename T>
__device__ __forceinline__ T from_float(float v) {
    if constexpr (std::is_same_v<T, half>) {
        return __float2half(v);
    } else if constexpr (std::is_same_v<T, __mt_bfloat16>) {
        return __float2bfloat16(v);
    } else {
        return static_cast<T>(v);
    }
}

// ================================================================
// Softsign Functor
// y = x / (1 + |x|)
// ================================================================
struct SoftsignOp {
    static constexpr size_t num_inputs = 1;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        float xf = to_float(x);
        float res = xf / (1.0f + fabsf(xf));
        return from_float<T>(res);
    }
};

// ================================================================
// TensorMetadata
// ================================================================
static constexpr int MAX_DIMS = 8;

struct TensorMetadata {
    int ndim;
    int64_t shape[MAX_DIMS];
    int64_t strides[MAX_DIMS];
};

// ================================================================
// Kernel 1: 连续内存
// ================================================================
template <typename T>
__global__ void softsign_kernel_contiguous(T *output, const T *input, size_t n) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        SoftsignOp functor;
        output[idx] = functor(input[idx]);
    }
}

// ================================================================
// Kernel 2: 非连续内存 (Strided)
// ================================================================
template <typename T>
__global__ void softsign_kernel_strided(
    T *output,
    const T *input,
    size_t n,
    TensorMetadata meta) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        size_t offset = 0;
        size_t t = idx;

#pragma unroll
        for (int d = meta.ndim - 1; d >= 0; --d) {
            size_t dim_size = meta.shape[d];
            size_t coord = t % dim_size;
            t /= dim_size;
            offset += coord * meta.strides[d];
        }

        SoftsignOp functor;
        output[idx] = functor(input[offset]);
    }
}

// ================================================================
// Launch Kernel
// ================================================================
template <typename T>
void launch_kernel(
    void *output,
    const void *input,
    const TensorMetadata &meta,
    size_t numel,
    bool is_contiguous,
    void *stream) {
    auto musa_stream = reinterpret_cast<musaStream_t>(stream);

    dim3 block(256);
    dim3 grid((numel + block.x - 1) / block.x);

    if (is_contiguous) {
        softsign_kernel_contiguous<T><<<grid, block, 0, musa_stream>>>(
            reinterpret_cast<T *>(output),
            reinterpret_cast<const T *>(input),
            numel);
    } else {
        softsign_kernel_strided<T><<<grid, block, 0, musa_stream>>>(
            reinterpret_cast<T *>(output),
            reinterpret_cast<const T *>(input),
            numel,
            meta);
    }
}

} // namespace op::softsign::moore

#endif // __SOFTSIGN_MOORE_KERNEL_H__
