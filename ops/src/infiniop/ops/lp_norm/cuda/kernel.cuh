#ifndef __LP_NORM_KERNEL_CUH__
#define __LP_NORM_KERNEL_CUH__

#include <cub/block/block_reduce.cuh>

template <typename T, unsigned int BLOCK_SIZE>
__device__ void blockLPNormKernel(
    T const *input, T *output, float p, size_t dimsize,
    ptrdiff_t stride, float eps) {

    int tid = blockIdx.x % stride + (blockIdx.x - blockIdx.x % stride) * dimsize; // now, tid = i(JKS) + k(S) + s;
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    float local_max = 0.0f;
    for (int ind = threadIdx.x; ind < dimsize; ind += BLOCK_SIZE) {
        local_max = max(local_max, fabsf((float)input[tid + ind * stride]));
    }
    __shared__ float global_max;
#if CUDART_VERSION >= 12090
    float max_block = BlockReduce(temp_storage).Reduce(local_max, ::cuda::maximum());
#elif defined(ENABLE_HYGON_API)
    float max_block = BlockReduce(temp_storage).Reduce(
        local_max, [](const float &a, const float &b) { return (a > b) ? a : b; }, BLOCK_SIZE);
#else
    float max_block = BlockReduce(temp_storage).Reduce(local_max, cub::Max());
#endif
    if (threadIdx.x == 0) { // must set threadIdx.x = 0 write the output to memory
        global_max = max_block;
    }
    __syncthreads();
    float global_max_inv = __fdividef(1.0F, max(global_max, eps));

    float p_partial = 0.0f;
    for (int ind = threadIdx.x; ind < dimsize; ind += BLOCK_SIZE) {
        p_partial += powf((float)input[tid + ind * stride] * global_max_inv, p);
    }

    __shared__ float p_total;
    float p_block = BlockReduce(temp_storage).Sum(p_partial);
    if (threadIdx.x == 0) { // must set threadIdx.x = 0 write the output to memory
        p_total = powf(p_block, 1.0f / p);
    }
    __syncthreads();
    float inv = __fdividef(1.0F, p_total + eps) * global_max_inv;

    for (int ind = threadIdx.x; ind < dimsize; ind += BLOCK_SIZE) {
        output[tid + ind * stride] = static_cast<T>(
            static_cast<float>(
                input[tid + ind * stride])
            * inv);
    }
}

template <typename T, unsigned int BLOCK_SIZE>
__device__ void blockLPNormStridesKernel(
    T const *input, T *output, const ptrdiff_t *output_strides,
    const ptrdiff_t *input_strides,
    const size_t *shape, int ndim, float p, size_t dimsize,
    float eps) {

    // 只能处理axis=-1
    int ind_i = 0; // input id
    int ind_o = 0; // output id
    int tid = blockIdx.x;
    for (int j = ndim - 2; j >= 0; j--) {
        ind_i += (tid % (int)shape[j]) * (int)input_strides[j];
        ind_o += (tid % (int)shape[j]) * (int)output_strides[j];
        tid = tid / (int)shape[j];
    }
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    float local_max = 0.0f;
    for (int ind = threadIdx.x; ind < dimsize; ind += BLOCK_SIZE) {
        local_max = max(local_max, fabsf((float)input[ind_i + ind]));
    }
    __shared__ float global_max;
#if CUDART_VERSION >= 12090
    float max_block = BlockReduce(temp_storage).Reduce(local_max, ::cuda::maximum());
#elif defined(ENABLE_HYGON_API)
    float max_block = BlockReduce(temp_storage).Reduce(
        local_max, [](const float &a, const float &b) { return (a > b) ? a : b; }, BLOCK_SIZE);
#else
    float max_block = BlockReduce(temp_storage).Reduce(local_max, cub::Max());
#endif
    if (threadIdx.x == 0) { // must set threadIdx.x = 0 write the output to memory
        global_max = max_block;
    }
    __syncthreads();
    float global_max_inv = __fdividef(1.0F, max(global_max, eps));

    float p_partial = 0.0f;
    for (int ind = threadIdx.x; ind < dimsize; ind += BLOCK_SIZE) {
        p_partial += powf((float)input[ind_i + ind] * global_max_inv, p);
    }

    __shared__ float p_total;
    float p_block = BlockReduce(temp_storage).Sum(p_partial);
    if (threadIdx.x == 0) { // must set threadIdx.x = 0 write the output to memory
        p_total = powf(p_block, 1.0f / p);
    }
    __syncthreads();
    float inv = __fdividef(1.0F, p_total + eps) * global_max_inv;

    for (int ind = threadIdx.x; ind < dimsize; ind += BLOCK_SIZE) {
        output[ind_o + ind] = static_cast<T>(
            static_cast<float>(
                input[ind_i + ind])
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

template <typename T, unsigned int BLOCK_SIZE_x, unsigned int BLOCK_SIZE_y>
__device__ void warpLPNormKernel(T const *input, T *output,
                                 float p, size_t othersize, size_t dimsize,
                                 ptrdiff_t stride, float eps) {
    int otherIdx = blockIdx.x * blockDim.y + threadIdx.y;

    int tid = otherIdx % stride + (otherIdx - otherIdx % stride) * dimsize;

    if (otherIdx < othersize) {

        __shared__ float p_total[BLOCK_SIZE_y];
        __shared__ float p_max[BLOCK_SIZE_y];
        float local_max = 0.0f;
        for (int ind = threadIdx.x; ind < dimsize; ind += BLOCK_SIZE_x) {
            local_max = max(local_max, fabsf((float)input[tid + ind * stride]));
        }
        local_max = WarpAllReduce<MaxOp, float, BLOCK_SIZE_x>(local_max);
        if (threadIdx.x == 0) {
            p_max[threadIdx.y] = local_max;
        }
        __syncthreads();
        float global_max = max(p_max[threadIdx.y], eps);
        float global_max_inv = __fdividef(1.0F, global_max);
        float p_data = 0.0f;

        for (int ind = threadIdx.x; ind < dimsize; ind += BLOCK_SIZE_x) {
            float v = fabsf((float)input[tid + ind * stride]) * global_max_inv;
            p_data += powf(v, p);
        }

        p_data = WarpAllReduce<SumOp, float, BLOCK_SIZE_x>(p_data);

        if (threadIdx.x == 0) {
            p_total[threadIdx.y] = powf(p_data, 1.0f / p);
        }
        __syncthreads();

        //--------------------------------------------
        float inv = __fdividef(1.0F, p_total[threadIdx.y] + eps) * global_max_inv;
        for (int ind = threadIdx.x; ind < dimsize; ind += BLOCK_SIZE_x) {
            output[tid + ind * stride] = static_cast<T>(
                (float)input[tid + ind * stride] * inv);
        }
    }
}

template <typename T, unsigned int BLOCK_SIZE_x, unsigned int BLOCK_SIZE_y>
__device__ void warpLPNormStridesKernel(T const *input, T *output, const ptrdiff_t *output_strides,
                                        const ptrdiff_t *input_strides,
                                        const size_t *shape, int ndim,
                                        float p, size_t othersize, size_t dimsize,
                                        float eps) {
    int ind_i = 0; // input id
    int ind_o = 0; // output id
    int tid = blockIdx.x * blockDim.y + threadIdx.y;

    if (tid < othersize) {
        for (int j = ndim - 2; j >= 0; j--) {
            ind_i += (tid % (int)shape[j]) * (int)input_strides[j];
            ind_o += (tid % (int)shape[j]) * (int)output_strides[j];
            tid = tid / (int)shape[j];
        }
        __shared__ float p_total[BLOCK_SIZE_y];
        __shared__ float p_max[BLOCK_SIZE_y];
        float local_max = 0.0f;
        for (int ind = threadIdx.x; ind < dimsize; ind += BLOCK_SIZE_x) {
            local_max = max(local_max, fabsf((float)input[ind_i + ind]));
        }
        local_max = WarpAllReduce<MaxOp, float, BLOCK_SIZE_x>(local_max);
        if (threadIdx.x == 0) {
            p_max[threadIdx.y] = local_max;
        }
        __syncthreads();
        float global_max = max(p_max[threadIdx.y], eps);
        float global_max_inv = __fdividef(1.0F, global_max);
        float p_data = 0.0f;

        for (int ind = threadIdx.x; ind < dimsize; ind += BLOCK_SIZE_x) {
            float v = fabsf((float)input[ind_i + ind]) * global_max_inv;
            p_data += powf(v, p);
        }

        p_data = WarpAllReduce<SumOp, float, BLOCK_SIZE_x>(p_data);

        if (threadIdx.x == 0) {
            p_total[threadIdx.y] = powf(p_data, 1.0f / p);
        }
        __syncthreads();

        //--------------------------------------------
        float inv = __fdividef(1.0F, p_total[threadIdx.y] + eps) * global_max_inv;
        for (int ind = threadIdx.x; ind < dimsize; ind += BLOCK_SIZE_x) {
            output[ind_o + ind] = static_cast<T>(
                (float)input[ind_i + ind] * inv);
        }
    }
}

#endif // __LP_NORM_KERNEL_CUH__
