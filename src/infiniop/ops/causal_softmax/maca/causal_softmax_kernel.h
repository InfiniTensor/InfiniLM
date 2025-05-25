#ifndef __CAUSAL_SOFTMAX_KERNEL_H__
#define __CAUSAL_SOFTMAX_KERNEL_H__

#include "../../../devices/maca/maca_kernel_common.h"
#include "../../../reduce/maca/reduce.h"

template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
INFINIOP_MACA_KERNEL causalSoftmax(
    Tdata *y_, const Tdata *x_,
    size_t batch, size_t height, size_t width,
    ptrdiff_t y_stride_b, ptrdiff_t y_stride_h,
    ptrdiff_t x_stride_b, ptrdiff_t x_stride_h) {

    Tdata *y = y_                       // threadIdx.x for col_id
             + blockIdx.y * y_stride_b  // gridDim.y for batch_id
             + blockIdx.x * y_stride_h; // gridDim.x for row_id
    const Tdata *x = x_ + blockIdx.y * x_stride_b + blockIdx.x * x_stride_h;

    // [Reduce] Find max value in each row and store in shared memory
    __shared__ Tdata max_;
    Tdata max_0 = op::common_maca::reduce_op::max<BLOCK_SIZE, Tdata>(x, width);
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
#ifdef ENABLE_MACA_API
            y[col] = exp_(x[col] - max_);
#else
            y[col] = exp(x[col] - max_);
#endif
        } else {
            y[col] = Tdata(0);
        }
    }
    __syncthreads();

    // [Reduce] Find the sum of each updated row and store in shared memory
    __shared__ Tcompute sum_;
    Tcompute sum_0 = op::common_maca::reduce_op::sum<BLOCK_SIZE, Tdata, Tcompute>(y, width);
    if (threadIdx.x == 0) {
        sum_ = sum_0;
    }
    __syncthreads();

    // [Elementwise] Divide each element by the sum and store in shared memory
    for (size_t col = threadIdx.x; col < width; col += BLOCK_SIZE) {
        y[col] /= Tdata(sum_);
    }
}

#endif // __CAUSAL_SOFTMAX_KERNEL_H__
