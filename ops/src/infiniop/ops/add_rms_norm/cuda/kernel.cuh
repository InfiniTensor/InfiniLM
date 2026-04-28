#ifndef __ADD_RMS_NORM_CUDA_KERNEL_H__
#define __ADD_RMS_NORM_CUDA_KERNEL_H__

#include <cub/block/block_reduce.cuh>

template <unsigned int BLOCK_SIZE, typename Tcompute, typename Tdata, typename Tweight>
__device__ void add_rmsnormBlock(
    Tdata *__restrict__ y,
    Tdata *__restrict__ residual_out,
    ptrdiff_t stride_y_batch,
    ptrdiff_t stride_y_nhead,
    ptrdiff_t stride_residual_out_batch,
    ptrdiff_t stride_residual_out_nhead,
    const Tdata *__restrict__ a,
    ptrdiff_t stride_a_batch,
    ptrdiff_t stride_a_nhead,
    const Tdata *__restrict__ b,
    ptrdiff_t stride_b_batch,
    ptrdiff_t stride_b_nhead,
    const Tweight *__restrict__ w,
    size_t nhead,
    size_t dim,
    float epsilon) {
    // Each block takes care of one head in one batch
    // Each thread deals with every block_size element in the row
    size_t batch_idx = blockIdx.x / nhead;
    size_t head_idx = blockIdx.x % nhead;

    auto y_ptr = y + batch_idx * stride_y_batch + head_idx * stride_y_nhead;
    auto a_ptr = a + batch_idx * stride_a_batch + head_idx * stride_a_nhead;
    auto b_ptr = b + batch_idx * stride_b_batch + head_idx * stride_b_nhead;
    auto w_ptr = w;
    Tdata *residual_out_ptr = residual_out + batch_idx * stride_residual_out_batch + head_idx * stride_residual_out_nhead;

    // Compute add(a, b) and sum of squares in one pass
    Tcompute sum_squared = 0;
    for (size_t i = threadIdx.x; i < dim; i += BLOCK_SIZE) {
        Tcompute sum_val = Tcompute(a_ptr[i]) + Tcompute(b_ptr[i]);
        residual_out_ptr[i] = Tdata(sum_val); // Store add result
        sum_squared += sum_val * sum_val;
    }

    // Block-reduce sum of squares
    using BlockReduce = cub::BlockReduce<Tcompute, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    sum_squared = BlockReduce(temp_storage).Sum(sum_squared);

    // Thread_0 computes RMS=1/sqrt(ss/dim+epsilon) and stores in shared memory
    __shared__ Tcompute rms;
    if (threadIdx.x == 0) {
        rms = Tcompute(rsqrtf(sum_squared / Tcompute(dim) + epsilon));
    }
    __syncthreads();

    // Apply normalization: y = (a + b) * w * rms
    // Reuse stored values from residual_out
    for (size_t i = threadIdx.x; i < dim; i += BLOCK_SIZE) {
        Tcompute sum_val = Tcompute(residual_out_ptr[i]); // Reuse stored value
        y_ptr[i] = Tdata(sum_val * Tcompute(w_ptr[i]) * rms);
    }
}

#endif
