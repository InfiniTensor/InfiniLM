#ifndef __CAUSAL_SOFTMAX_KERNEL_CUH__
#define __CAUSAL_SOFTMAX_KERNEL_CUH__

#include "../../../devices/cuda/cuda_common.cuh"

template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
INFINIOP_CUDA_KERNEL causalSoftmax(Tdata *data_, size_t batch, size_t height, size_t width, ptrdiff_t stride_b, ptrdiff_t stride_h) {
    Tdata *data = data_                  // threadIdx.x for col_id
                + blockIdx.y * stride_b  // gridDim.y for batch_id
                + blockIdx.x * stride_h; // gridDim.x for row_id

    // [Reduce] Find max value in each row and store in shared memory
    __shared__ Tdata max_;
    Tdata max_0 = op::common_cuda::reduce_op::max<BLOCK_SIZE, Tdata>(data, width);
    if (threadIdx.x == 0) {
        max_ = max_0;
    }
    __syncthreads();

    // [Elementwise] Subtract max value from each element and apply causal mask
    for (size_t col = threadIdx.x; col < width; col += BLOCK_SIZE) {
        //   row_id ↓ |<-     width   ->|
        //          0 | * * * ... *     |
        //          1 | * * * ... * *   |
        //          2 | * * * ... * * * |
        //  height: 3  col_id->
        if (width + blockIdx.x >= threadIdx.x + height) {
            data[col] = exp(data[col] - max_);
        } else {
            data[col] = Tdata(0);
        }
    }
    __syncthreads();

    // [Reduce] Find the sum of each updated row and store in shared memory
    __shared__ Tcompute sum_;
    Tcompute sum_0 = op::common_cuda::reduce_op::sum<BLOCK_SIZE, Tdata, Tcompute>(data, width);
    if (threadIdx.x == 0) {
        sum_ = sum_0;
    }
    __syncthreads();

    // [Elementwise] Divide each element by the sum and store in shared memory
    for (size_t col = threadIdx.x; col < width; col += BLOCK_SIZE) {
        data[col] /= Tdata(sum_);
    }
}

#endif // __CAUSAL_SOFTMAX_KERNEL_CUH__
