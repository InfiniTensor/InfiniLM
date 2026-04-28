#ifndef __TOPK_CUDA_KERNEL_CUH__
#define __TOPK_CUDA_KERNEL_CUH__

#include <cmath> // NAN
#include <cub/block/block_radix_sort.cuh>
#include <stdint.h>

namespace op::topk::cuda {
__forceinline__ __device__ __host__ size_t baseOffsetExcludingDim(
    size_t flat_row,
    size_t ndim,
    const size_t *shape,
    const ptrdiff_t *strides,
    size_t dim) {
    size_t res = 0;
    for (size_t i = ndim; i-- > 0;) {
        if (i == dim) {
            continue;
        }
        res += (flat_row % shape[i]) * strides[i];
        flat_row /= shape[i];
    }
    return res;
}

__forceinline__ __device__ __host__ size_t indexToOffset(
    size_t flat_index,
    size_t ndim,
    const size_t *shape,
    const ptrdiff_t *strides) {
    size_t res = 0;
    for (size_t i = ndim; i-- > 0;) {
        res += (flat_index % shape[i]) * strides[i];
        flat_index /= shape[i];
    }
    return res;
}

template <typename Tdata>
__device__ __forceinline__ float to_float(Tdata v);

template <>
__device__ __forceinline__ float to_float<float>(float v) { return v; }

template <>
__device__ __forceinline__ float to_float<half>(half v) { return __half2float(v); }

#if defined(ENABLE_MOORE_API)
using bf16_t = __mt_bfloat16;
#elif defined(ENABLE_METAX_API)
using bf16_t = __hpcc_bfloat16;
#else
// CUDA / NVIDIA / ILUVATAR
using bf16_t = __nv_bfloat16;
#endif
template <>
__device__ __forceinline__ float to_float<bf16_t>(bf16_t v) {
    return __bfloat162float(v);
}

// float -> ordered uint32
__device__ __forceinline__ uint32_t float_to_uint_ordered(float value) {
    uint32_t bits = *reinterpret_cast<uint32_t *>(&value);
    uint32_t mask = (uint32_t)(-((int32_t)bits >> 31)) | 0x80000000u;
    return bits ^ mask;
}

template <typename Tdata>
__global__ void gather_rowwise(const Tdata *input, uint32_t *cur_vals, int32_t *cur_idx,
                               size_t rows, size_t n, size_t ndim, size_t dim, const size_t *shape, const ptrdiff_t *strides) {
    size_t row = blockIdx.y;
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (row >= rows || i >= n) {
        return;
    }
    size_t base = baseOffsetExcludingDim(row, ndim, shape, strides, dim);
    size_t off = base + i * strides[dim];
    cur_vals[row * n + i] = float_to_uint_ordered(to_float<Tdata>(input[off]));
    cur_idx[row * n + i] = i;
}

__global__ void init_row_state(int32_t *cur_n, int32_t *rem_k, int32_t *out_pos, size_t rows, size_t n, size_t k) {
    int32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows) {
        cur_n[r] = n;
        rem_k[r] = k;
        out_pos[r] = 0;
    }
}

__global__ void zero_row_counters(int32_t *ones_count, int32_t *zeros_count, size_t rows) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows) {
        ones_count[r] = 0;
        zeros_count[r] = 0;
    }
}

template <size_t BLOCK_SIZE>
__global__ void partition_rowwise(const uint32_t *cur_vals, int32_t *cur_idx, uint32_t *ones_vals, int32_t *ones_idx,
                                  uint32_t *zeros_vals, int32_t *zeros_idx, const int32_t *cur_n, size_t rows, size_t n,
                                  int32_t bit_pos, bool largest, int32_t *ones_count, int32_t *zeros_count) {
    int32_t row = blockIdx.y;
    if (row >= rows) {
        return;
    }

    __shared__ uint32_t sh1_vals[BLOCK_SIZE];
    __shared__ int32_t sh1_idx[BLOCK_SIZE];
    __shared__ uint32_t sh0_vals[BLOCK_SIZE];
    __shared__ int32_t sh0_idx[BLOCK_SIZE];
    __shared__ int sh1_n, sh0_n;
    __shared__ int32_t base1, base0;

    int32_t tid = threadIdx.x;
    if (tid == 0) {
        sh1_n = 0;
        sh0_n = 0;
    }
    __syncthreads();

    int32_t i = blockIdx.x * blockDim.x + tid;
    int32_t cn = cur_n[row];
    if (i < cn) {
        int32_t off = row * n + i;
        int32_t idx = cur_idx[off];
        uint32_t key = cur_vals[off];
        uint32_t cmp_key = largest ? key : ~key;
        int32_t b = (cmp_key >> bit_pos) & 1;

        if (b) {
            int32_t p = atomicAdd(&sh1_n, 1);
            sh1_vals[p] = key;
            sh1_idx[p] = idx;
        } else {
            int32_t p = atomicAdd(&sh0_n, 1);
            sh0_vals[p] = key;
            sh0_idx[p] = idx;
        }
    }
    __syncthreads();

    if (tid == 0) {
        base1 = atomicAdd(&ones_count[row], sh1_n);
        base0 = atomicAdd(&zeros_count[row], sh0_n);
    }
    __syncthreads();

    for (int32_t j = tid; j < sh1_n; j += blockDim.x) {
        int32_t o = row * n + base1 + j;
        ones_vals[o] = sh1_vals[j];
        ones_idx[o] = sh1_idx[j];
    }
    for (int32_t j = tid; j < sh0_n; j += blockDim.x) {
        int32_t o = row * n + base0 + j;
        zeros_vals[o] = sh0_vals[j];
        zeros_idx[o] = sh0_idx[j];
    }
}

template <size_t BLOCK_SIZE>
__global__ void decide_and_compact(uint32_t *cur_vals, int32_t *cur_idx, const uint32_t *ones_vals, const int32_t *ones_idx, const uint32_t *zeros_vals, const int32_t *zeros_idx,
                                   const int32_t *ones_count, const int32_t *zeros_count, int32_t *cur_n, int32_t *rem_k, int32_t *out_pos,
                                   uint32_t *sel_vals, int32_t *sel_idx, size_t rows, size_t n, size_t k) {
    int32_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    int32_t tid = threadIdx.x;
    int32_t rem = rem_k[row];
    if (rem <= 0) {
        return;
    }
    int32_t oc = ones_count[row];
    int32_t zc = zeros_count[row];
    int32_t pos = out_pos[row];

    bool keep_ones = (oc >= rem);
    if (!keep_ones) {
        for (int32_t j = tid; j < oc; j += blockDim.x) {
            if (pos + j < k) {
                int32_t o = row * n + j;
                sel_vals[row * k + pos + j] = ones_vals[o];
                sel_idx[row * k + pos + j] = ones_idx[o];
            }
        }
    }
    __syncthreads();
    if (tid == 0) {
        if (keep_ones) {
            cur_n[row] = oc;
        } else {
            out_pos[row] = pos + oc;
            rem_k[row] = rem - oc;
            cur_n[row] = zc;
        }
    }
    __syncthreads();
    int32_t new_n = cur_n[row];
    for (int32_t j = tid; j < new_n; j += blockDim.x) {
        int32_t o = row * n + j;
        cur_vals[o] = keep_ones ? ones_vals[o] : zeros_vals[o];
        cur_idx[o] = keep_ones ? ones_idx[o] : zeros_idx[o];
    }
}

template <size_t BLOCK_SIZE>
__global__ void take_remaining(const uint32_t *cur_vals, const int32_t *cur_idx, const int32_t *cur_n, const int32_t *rem_k, const int32_t *out_pos,
                               uint32_t *sel_vals, int32_t *sel_idx, size_t rows, size_t n, size_t k) {
    int32_t row = blockIdx.x;
    int32_t tid = threadIdx.x;
    if (row >= rows) {
        return;
    }
    int32_t rem = rem_k[row];
    int32_t pos = out_pos[row];
    int32_t cn = cur_n[row];

    int32_t take = rem;
    if (take > cn) {
        take = cn;
    }
    for (int32_t j = tid; j < take; j += blockDim.x) {
        if (pos + j < k) {
            int32_t o = row * k + pos + j;
            sel_vals[o] = cur_vals[row * n + j];
            sel_idx[o] = cur_idx[row * n + j];
        }
    }
}

template <typename Tdata>
__global__ void scatter_to_output(const Tdata *input, const int32_t *sel_idx, Tdata *values_out, int32_t *indices_out,
                                  size_t rows, size_t k, size_t ndim, size_t dim, const size_t *input_shape, const ptrdiff_t *input_strides,
                                  const size_t *output_shape, const ptrdiff_t *output_strides) {
    int32_t row = blockIdx.y;
    int32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || j >= k) {
        return;
    }

    int32_t output_base = baseOffsetExcludingDim(row, ndim, output_shape, output_strides, dim);
    int32_t output_off = output_base + j * output_strides[dim];
    int32_t input_base = baseOffsetExcludingDim(row, ndim, input_shape, input_strides, dim);
    int32_t input_off = input_base + sel_idx[row * k + j] * input_strides[dim];

    values_out[output_off] = input[input_off];
    indices_out[output_off] = sel_idx[row * k + j];
}

} // namespace op::topk::cuda

#endif // __TOPK_CUDA_KERNEL_H__
