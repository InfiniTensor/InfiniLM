#ifndef __CROSS_ENTROPY_KERNEL_CUH__
#define __CROSS_ENTROPY_KERNEL_CUH__

#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../reduce/cuda/reduce.cuh"

template <unsigned int BLOCK_SIZE, typename Tdata, typename Tidx, typename Tcompute = float>
__device__ void crossEntropyKernel(
    Tdata *y_,
    const Tdata *x_,
    const void *target_,
    size_t outer_size,
    size_t vocab_size,
    ptrdiff_t x_stride) {

    size_t row_idx = blockIdx.x;
    if (row_idx >= outer_size) {
        return;
    }

    const Tdata *x = x_ + row_idx * x_stride;
    const Tidx *target = reinterpret_cast<const Tidx *>(target_);

    Tidx label = target[row_idx];

    Tdata max_val_raw = op::common_cuda::reduce_op::max<BLOCK_SIZE, Tdata>(x, vocab_size);
    __shared__ Tcompute max_val_shared;
    if (threadIdx.x == 0) {
        max_val_shared = static_cast<Tcompute>(max_val_raw);
    }
    __syncthreads();
    Tcompute max_val = max_val_shared;

    Tcompute thread_sum = 0.0f;
    for (size_t col = threadIdx.x; col < vocab_size; col += BLOCK_SIZE) {
        Tcompute val = static_cast<Tcompute>(x[col]);
        thread_sum += expf(val - max_val);
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    static __shared__ Tcompute shared_sum[32];
    int lane = threadIdx.x % warpSize;
    int warp = threadIdx.x / warpSize;

    if (lane == 0) {
        shared_sum[warp] = thread_sum;
    }
    __syncthreads();

    Tcompute block_sum = 0.0f;
    if (warp == 0) {

        if (lane < (BLOCK_SIZE + warpSize - 1) / warpSize) {
            block_sum = shared_sum[lane];
        }
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
    }

    if (threadIdx.x == 0) {
        Tcompute log_term = logf(block_sum) + max_val;

        Tcompute target_logit = 0.0f;

        if (label >= 0 && static_cast<size_t>(label) < vocab_size) {
            target_logit = static_cast<Tcompute>(x[label]);
        } else {

            log_term = 0.0f;
        }

        y_[row_idx] = static_cast<Tdata>(log_term - target_logit);
    }
}

#endif
