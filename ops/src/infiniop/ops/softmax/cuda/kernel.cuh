#ifndef __SOFTMAX_KERNEL_CUH__
#define __SOFTMAX_KERNEL_CUH__

#include <cub/block/block_reduce.cuh>

struct __align__(8) DataMaxSum { // update the global max and sum, store the
                                 // output at max_tmp and sum_tmp
    float max_tmp;               // store max
    float sum_tmp;               // store sum
};
__device__ __forceinline__ DataMaxSum reduce_dms_op(DataMaxSum a,
                                                    DataMaxSum b) {
    bool a_bigger = (a.max_tmp > b.max_tmp);
    DataMaxSum bigger = a_bigger ? a : b;
    DataMaxSum smaller = a_bigger ? b : a;
    bigger.sum_tmp = bigger.sum_tmp + smaller.sum_tmp * __expf(smaller.max_tmp - bigger.max_tmp);

    return bigger;
}
template <typename T, unsigned int BLOCK_SIZE>
__device__ void blockSoftmaxKernel(
    T const *input, T *output, size_t dimsize,
    ptrdiff_t stride) {

    int tid = blockIdx.x % stride + (blockIdx.x - blockIdx.x % stride) * dimsize; // now, tid = i(JKS) + k(S) + s;

    DataMaxSum dms_partial;
    dms_partial.max_tmp = -__FLT_MAX__;
    dms_partial.sum_tmp = 0.0f;
    DataMaxSum dms_input;
    for (int ind = threadIdx.x; ind < dimsize; ind += BLOCK_SIZE) {
        dms_input.max_tmp = static_cast<float>(input[tid + ind * stride]);

        dms_input.sum_tmp = 1.0f;
        dms_partial = reduce_dms_op(dms_partial,
                                    dms_input); // reduce the data to one block
    }

    typedef cub::BlockReduce<DataMaxSum, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ DataMaxSum dms_total;
    DataMaxSum dms_block = BlockReduce(temp_storage).Reduce(dms_partial, reduce_dms_op);
    if (threadIdx.x == 0) { // must set threadIdx.x = 0 write the output to memory
        dms_total = dms_block;
    }
    __syncthreads();
    float inv = __fdividef(1.0F, dms_total.sum_tmp);

    for (int ind = threadIdx.x; ind < dimsize; ind += BLOCK_SIZE) {
        output[tid + ind * stride] = static_cast<T>(
            __expf(static_cast<float>(
                       input[tid + ind * stride])
                   - dms_total.max_tmp)
            * inv);
    }
}

template <typename T>
struct SumOp {
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return a + b;
    }
};

template <typename T>
struct MaxOp {
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return max(a, b);
    }
};
template <template <typename> class ReductionOp, typename T,
          int thread_group_width>
__inline__ __device__ T WarpAllReduce(T val) {
    for (int mask = thread_group_width / 2; mask > 0; mask /= 2) {
        val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

template <typename T, unsigned int BLOCK_SIZE_x, unsigned int BLOCK_SIZE_y, int numPerThreadx>
__device__ void warpSoftmaxKernel(T const *input, T *output,
                                  size_t othersize, size_t dimsize, ptrdiff_t stride) {
    int otherIdx = blockIdx.x * blockDim.y + threadIdx.y;

    int tid = otherIdx % stride + (otherIdx - otherIdx % stride) * dimsize;
    float dataPerThreadx[numPerThreadx];
    if (otherIdx < othersize) {

        __shared__ float max_total[BLOCK_SIZE_y];
        __shared__ float sum_total[BLOCK_SIZE_y];
        float max_data = -__FLT_MAX__;

        for (int ph = 0; threadIdx.x + ph * BLOCK_SIZE_x < dimsize; ph++) {
            dataPerThreadx[ph] = static_cast<float>(input[tid + (threadIdx.x + ph * BLOCK_SIZE_x) * stride]);
            max_data = max(max_data, dataPerThreadx[ph]);
        }

        max_data = WarpAllReduce<MaxOp, float, BLOCK_SIZE_x>(max_data);

        if (threadIdx.x == 0) {
            max_total[threadIdx.y] = max_data;
        }

        //--------------------------------------------
        float sum_data = 0.0f;

        for (int ph = 0; threadIdx.x + ph * BLOCK_SIZE_x < dimsize; ph++) {
            dataPerThreadx[ph] = __expf(dataPerThreadx[ph] - max_total[threadIdx.y]);
            sum_data += dataPerThreadx[ph];
        }

        sum_data = WarpAllReduce<SumOp, float, BLOCK_SIZE_x>(sum_data);

        if (threadIdx.x == 0) {
            sum_total[threadIdx.y] = sum_data;
        }

        //--------------------------------------------

        for (int ph = 0; threadIdx.x + ph * BLOCK_SIZE_x < dimsize; ph++) {
            output[tid + (threadIdx.x + ph * BLOCK_SIZE_x) * stride] = static_cast<T>(
                dataPerThreadx[ph] * __fdividef(1.0F, sum_total[threadIdx.y]));
        }
    }
}

#endif // __SOFTMAX_KERNEL_CUH__
