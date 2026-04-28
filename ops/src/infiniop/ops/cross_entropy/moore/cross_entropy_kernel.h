#ifndef __CROSS_ENTROPY_KERNEL_CUH__
#define __CROSS_ENTROPY_KERNEL_CUH__

template <unsigned int BLOCK_SIZE, typename Tdata, typename Tidx, typename Tcompute>
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

    Tcompute thread_sum = Tcompute(0);
    for (size_t col = threadIdx.x; col < vocab_size; col += BLOCK_SIZE) {
        Tcompute val = static_cast<Tcompute>(x[col]);
        thread_sum += expf(val - max_val);
    }

    using BlockReduce = cub::BlockReduce<Tcompute, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    Tcompute block_sum = BlockReduce(temp_storage).Sum(thread_sum);

    if (threadIdx.x == 0) {
        if (label < 0 || static_cast<size_t>(label) >= vocab_size) {
            y_[row_idx] = static_cast<Tdata>(0.0f);
            return;
        }
        Tcompute log_term = logf(block_sum) + max_val;
        Tcompute target_logit = static_cast<Tcompute>(x[label]);
        y_[row_idx] = static_cast<Tdata>(log_term - target_logit);
    }
}

#endif
