// Custom HIP GEMV kernel for the small-M (M=1) bs=1-decode hot path on
// Hygon DCU. The DTK hipBLAS heuristic picks `MT16x16x32` for these shapes
// which runs at ~10 GFLOPS / 0.02% of peak. A purpose-built GEMV gets us
// to bandwidth-bound territory.
//
// Shape pattern (verified by diag in gemm_nvidia.cu calculate()):
//   y[M]      = alpha * (W[M,K] @ x[K]) + beta * y[M]
//   W: M×K row-major, lda=K
//   x: contiguous K-vector
//   y: contiguous M-vector
//   dtype: BF16 IO, FP32 accumulator
//
// The cuBLAS call equivalent (after `cublasGemmStridedBatchedEx`'s view):
//   M=N(output_dim), N_cublas=1, K=in_dim, op_a=T (=> A row-major M×K), op_b=N
// We are called with `n=1` so this kernel matches every linear in decode.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace op::gemm::nvidia {

// One block per output row m. WARP_SIZE threads (one wavefront on DCU = 64).
// Each thread strides through the K dim, accumulates partial dot product,
// then warp-reduces via shuffle. Bandwidth-bound for K ≥ 256.
template <int WARP_SIZE>
__global__ void gemv_bf16_kernel(
    const __nv_bfloat16 *__restrict__ A,  // M×K row-major, lda=K
    const __nv_bfloat16 *__restrict__ x,  // K
    __nv_bfloat16 *__restrict__ y,        // M
    int K, int M,
    float alpha, float beta) {
    int m = blockIdx.x;
    if (m >= M) return;
    int tid = threadIdx.x;

    const __nv_bfloat16 *A_row = A + (size_t)m * K;

    float acc = 0.0f;
    // Vectorize loads as bf16x4 (= 8 bytes) to saturate HBM channels.
    int k = tid * 4;
    for (; k + 3 < K; k += WARP_SIZE * 4) {
        __nv_bfloat162 a01 = *reinterpret_cast<const __nv_bfloat162 *>(A_row + k);
        __nv_bfloat162 a23 = *reinterpret_cast<const __nv_bfloat162 *>(A_row + k + 2);
        __nv_bfloat162 x01 = *reinterpret_cast<const __nv_bfloat162 *>(x + k);
        __nv_bfloat162 x23 = *reinterpret_cast<const __nv_bfloat162 *>(x + k + 2);
        acc += __bfloat162float(a01.x) * __bfloat162float(x01.x);
        acc += __bfloat162float(a01.y) * __bfloat162float(x01.y);
        acc += __bfloat162float(a23.x) * __bfloat162float(x23.x);
        acc += __bfloat162float(a23.y) * __bfloat162float(x23.y);
    }
    // Tail (K not a multiple of WARP_SIZE*4)
    for (int kt = (K & ~(WARP_SIZE * 4 - 1)) + tid; kt < K; kt += WARP_SIZE) {
        acc += __bfloat162float(A_row[kt]) * __bfloat162float(x[kt]);
    }

    // Warp reduce (DCU wavefront width = 64; HIP __shfl_xor handles wave-wide reduction).
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_xor(acc, offset);
    }

    if (tid == 0) {
        float old = beta == 0.0f ? 0.0f : __bfloat162float(y[m]);
        y[m] = __float2bfloat16(alpha * acc + beta * old);
    }
}

inline cudaError_t launch_gemv_bf16(
    const __nv_bfloat16 *A, const __nv_bfloat16 *x, __nv_bfloat16 *y,
    int K, int M,
    float alpha, float beta,
    cudaStream_t stream) {
    constexpr int WARP_SIZE = 64; // DCU wavefront
    dim3 grid(M);
    dim3 block(WARP_SIZE);
    gemv_bf16_kernel<WARP_SIZE><<<grid, block, 0, stream>>>(A, x, y, K, M, alpha, beta);
    return cudaGetLastError();
}

} // namespace op::gemm::nvidia
