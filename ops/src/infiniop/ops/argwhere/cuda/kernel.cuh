#ifndef ARGWHERE_NVIDIA_KERNEL_H
#define ARGWHERE_NVIDIA_KERNEL_H
#include <cstddef>
#include <cstdint>

__device__ void Index2Pos(size_t index, size_t ndim, const size_t *shapes,
                          size_t *pos) {
    for (int i = ndim - 1; i >= 0; i--) {
        pos[i] = index % shapes[i];
        index /= shapes[i];
    }
}
__device__ size_t pos2dest(size_t *pos, size_t ndim, const ptrdiff_t *strides) {
    size_t dest = 0;
    for (size_t i = 0; i < ndim; i++) {
        dest += pos[i] * strides[i];
    }
    return dest;
}

template <typename T>
__global__ void parallel_block_argwhere_kernel(T *data, int64_t *results,
                                               size_t N, const size_t *shapes, const ptrdiff_t *strides, size_t ndim, size_t *count) {
    extern __shared__ char smem[];
    size_t *tmp = reinterpret_cast<size_t *>(smem);

    size_t pos1[5], pos2[5]; // 两个数的在tensor中的索引
    bool is_zero1 = false, is_zero2 = false;
    int tid = threadIdx.x;
    int leaf_num = blockDim.x * 2; // equals to length of tmp

    if (tid * 2 < N) {
        Index2Pos(tid * 2, ndim, shapes, pos1);
        is_zero1 = fabs(data[pos2dest(pos1, ndim, strides)]) <= 1e-5;
        tmp[tid * 2] = !is_zero1;
    }
    if (tid * 2 + 1 < N) {
        Index2Pos(tid * 2 + 1, ndim, shapes, pos2);
        is_zero2 = fabs(data[pos2dest(pos2, ndim, strides)]) <= 1e-5;
        tmp[tid * 2 + 1] = !is_zero2;
    }

    __syncthreads();

    int offset = 1;
    for (int d = leaf_num >> 1; d > 0; d >>= 1) {
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            tmp[bi] += tmp[ai];
        }
        offset *= 2;
        __syncthreads();
    }

    if (tid == 0) {
        tmp[leaf_num - 1] = 0;
    }
    __syncthreads();

    for (int d = 1; d < leaf_num; d *= 2) {
        offset >>= 1;
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;

            int v = tmp[ai];
            tmp[ai] = tmp[bi];
            tmp[bi] += v;
        }
        __syncthreads();
    }

    // 写入最终结果
    if (!is_zero1 && tid * 2 < N) {
        for (int i = 0; i < ndim; i++) {
            results[tmp[2 * tid] * ndim + i] = pos1[i];
        }
    }
    if (!is_zero2 && tid * 2 + 1 < N) {
        for (int i = 0; i < ndim; i++) {
            results[tmp[2 * tid + 1] * ndim + i] = pos2[i];
        }
    }
    if (tid == blockDim.x - 1) {
        // printf("blockIdxDim = %d\n", blockDim.x);
        *count = tmp[N - 1] + (N == blockDim.x * 2 ? 1 : tmp[N] != tmp[N - 1]);
        // printf("finally: count = %d\n", tmp[leaf_num - 1]);
    }
}

template <typename T>
__global__ void parallel_block_scan_kernel(size_t N, int64_t *pre_sum) {
    // single block scan
    extern __shared__ char smem[];
    int64_t *tmp = reinterpret_cast<int64_t *>(smem);

    int tid = threadIdx.x;
    int leaf_num = blockDim.x * 2; // equals to length of tmp

    tmp[tid * 2] = tid * 2 < N - 1 ? pre_sum[tid * 2] : 0;
    tmp[tid * 2 + 1] = tid * 2 + 1 < N - 1 ? pre_sum[tid * 2 + 1] : 0;
    __syncthreads();

    int offset = 1;
    for (int d = leaf_num >> 1; d > 0; d >>= 1) {
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            tmp[bi] += tmp[ai];
        }
        offset *= 2;
        __syncthreads();
    }
    if (tid == 0) {
        tmp[leaf_num - 1] = 0;
    }
    __syncthreads();
    for (int d = 1; d < leaf_num; d <<= 1) {
        offset >>= 1;
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;

            int v = tmp[ai];
            tmp[ai] = tmp[bi];
            tmp[bi] += v;
        }
        __syncthreads();
    }
    if (tid * 2 < N) {
        pre_sum[tid * 2] = tmp[tid * 2];
    }
    if (tid * 2 + 1 < N) {
        pre_sum[tid * 2 + 1] = tmp[tid * 2 + 1];
    }
}

template <typename T>
__global__ void
parallel_large_argwhere_kernel(const T *data, int64_t *block_sum,
                               int64_t *results, size_t N, const size_t *shapes,
                               const ptrdiff_t *strides, size_t ndim) {
    // To be implemented for large N
    extern __shared__ int64_t tmp_argwhere[];
    int bid = blockIdx.x, tid = threadIdx.x;
    size_t pos1[5], pos2[5]; // 两个数的在tensor中的索引
    bool is_zero1 = false, is_zero2 = false;
    int block_offset = bid * blockDim.x * 2, leaf_num = blockDim.x * 2;
    tmp_argwhere[2 * tid] = tmp_argwhere[2 * tid + 1] = 0;
    if (block_offset + tid * 2 < N) {
        Index2Pos(block_offset + tid * 2, ndim, shapes, pos1);
        is_zero1 = fabs(data[pos2dest(pos1, ndim, strides)]) <= 1e-9;
        tmp_argwhere[2 * tid] = !is_zero1;
    }
    if (block_offset + tid * 2 + 1 < N) {
        Index2Pos(block_offset + tid * 2 + 1, ndim, shapes, pos2);
        is_zero2 = fabs(data[pos2dest(pos2, ndim, strides)]) <= 1e-9;
        tmp_argwhere[2 * tid + 1] = !is_zero2;
    }
    __syncthreads();
    int offset = 1;
    for (int d = leaf_num >> 1; d > 0; d >>= 1) {
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            tmp_argwhere[bi] += tmp_argwhere[ai];
        }
        offset *= 2;
        __syncthreads();
    }
    if (tid == 0) {
        block_sum[bid] = tmp_argwhere[leaf_num - 1];
        // printf("tmp_argwhere[%d] = %lld\n", leaf_num - 1,
        //        tmp_argwhere[leaf_num - 1]);
        // printf("block_sum[%d] = %lld\n", bid, block_sum[bid]);
        tmp_argwhere[leaf_num - 1] = 0;
    }
    __syncthreads();
    for (int d = 1; d < leaf_num; d <<= 1) {
        offset >>= 1;
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            int v = tmp_argwhere[ai];
            tmp_argwhere[ai] = tmp_argwhere[bi];
            tmp_argwhere[bi] += v;
        }
        __syncthreads();
    }

    if (!is_zero1 && block_offset + tid * 2 < N) {
        for (int i = 0; i < ndim; i++) {
            results[(tmp_argwhere[2 * tid] + block_offset) * ndim + i] = pos1[i];
        }
    }
    if (!is_zero2 && block_offset + tid * 2 + 1 < N) {
        for (int i = 0; i < ndim; i++) {
            results[(tmp_argwhere[2 * tid + 1] + block_offset) * ndim + i] = pos2[i];
        }
    }
}

__global__ void add_block_offset_kernel(int64_t *results, int64_t *tmp,
                                        int64_t *block_sums, size_t ndim) {
    int bid = blockIdx.x, tid = threadIdx.x;
    size_t block_offset = block_sums[bid], origin_offset = bid * blockDim.x * 2;
    if (2 * tid < block_sums[bid + 1] - block_sums[bid]) {
        for (int i = 0; i < ndim; i++) {
            results[(block_offset + 2 * tid) * ndim + i] = tmp[(origin_offset + tid * 2) * ndim + i];
        }
    }
    if (2 * tid + 1 < block_sums[bid + 1] - block_sums[bid]) {
        for (int i = 0; i < ndim; i++) {
            results[(block_offset + 2 * tid + 1) * ndim + i] = tmp[(origin_offset + tid * 2 + 1) * ndim + i];
        }
    }
}
#endif
