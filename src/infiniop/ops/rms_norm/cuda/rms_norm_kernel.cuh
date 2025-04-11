#ifndef __RMS_NORM_CUDA_KERNEL_H__
#define __RMS_NORM_CUDA_KERNEL_H__

#include "../../../devices/cuda/cuda_kernel_common.cuh"
#include "../../../reduce/cuda/reduce.cuh"

template <unsigned int BLOCK_SIZE, typename Tdata, typename Tweight, typename Tcompute>
INFINIOP_CUDA_KERNEL rmsnormBlock(
    Tdata *__restrict__ y,
    ptrdiff_t stride_y,
    const Tdata *__restrict__ x,
    ptrdiff_t stride_x,
    const Tweight *__restrict__ w,
    size_t dim,
    float epsilon) {
    // Each block takes care of a row of continuous data of length dim
    // Each thread deals with every block_size element in the row
    auto y_ptr = y + blockIdx.x * stride_y;
    auto x_ptr = x + blockIdx.x * stride_x;
    auto w_ptr = w;

    // Block-reduce sum of x^2
    Tcompute ss = op::common_cuda::reduce_op::sumSquared<BLOCK_SIZE, Tdata, Tcompute>(x_ptr, dim);

    // Thread_0 computes RMS=1/sqrt(ss/dim+epsilon) and stores in shared memory
    __shared__ Tcompute rms;
    if (threadIdx.x == 0) {
        rms = Tdata(rsqrtf(ss / Tcompute(dim) + epsilon));
    }
    __syncthreads();

    for (size_t i = threadIdx.x; i < dim; i += BLOCK_SIZE) {
        y_ptr[i] = Tdata(Tcompute(x_ptr[i]) * Tcompute(w_ptr[i]) * rms);
    }
}

#endif
