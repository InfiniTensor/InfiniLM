#ifndef ARGWHERE_MOORE_KERNEL_H
#define ARGWHERE_MOORE_KERNEL_H

__device__ void Index2Pos(size_t index, size_t ndim, const size_t *shapes,
                          size_t *pos) {
    for (int i = ndim - 1; i >= 0; i--) {
        pos[i] = index % shapes[i];
        index /= shapes[i];
    }
}
__device__ size_t pos2dest(size_t *pos, size_t ndim, const size_t *shapes,
                           const ptrdiff_t *strides) {
    size_t dest = 0;
    for (size_t i = 0; i < ndim; i++) {
        dest += pos[i] * strides[i];
    }
    return dest;
}
template <typename T>
__global__ void parallel_block_argwhere_kernel(T *data, int64_t *results,
                                               size_t N, const size_t *shapes,
                                               const ptrdiff_t *strides,
                                               size_t ndim, size_t *count) {
    extern __shared__ size_t tmp[];
    size_t pos1[5], pos2[5]; // 两个数的在tensor中的索引
    bool is_zero1 = false, is_zero2 = false;
    int tid = threadIdx.x;
    int leaf_num = blockDim.x * 2;

    if (tid * 2 < N) {
        Index2Pos(tid * 2, ndim, shapes, pos1);
        is_zero1 = fabs(data[pos2dest(pos1, ndim, shapes, strides)]) <= 1e-5;
        tmp[tid * 2] = !is_zero1;
    } else {
        tmp[tid * 2] = 0;
    }

    if (tid * 2 + 1 < N) {
        Index2Pos(tid * 2 + 1, ndim, shapes, pos2);
        is_zero2 = fabs(data[pos2dest(pos2, ndim, shapes, strides)]) <= 1e-5;
        tmp[tid * 2 + 1] = !is_zero2;
    } else {
        tmp[tid * 2 + 1] = 0;
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
        *count = tmp[N - 1] + (N == blockDim.x * 2 ? 1 : tmp[N] != tmp[N - 1]);
    }
}

#endif
