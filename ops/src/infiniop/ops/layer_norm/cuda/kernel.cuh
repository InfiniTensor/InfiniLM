#ifndef __LAYER_NORM_KERNEL_CUH__
#define __LAYER_NORM_KERNEL_CUH__
#include <cub/block/block_reduce.cuh>

template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
__device__ void layerNormKernel(
    Tdata *output,
    Tdata *input_standardization,
    Tdata *input_std_deviation,
    const Tdata *input,
    const Tdata *weight,
    const Tdata *bias,
    float eps,
    size_t normalized_size,
    const ptrdiff_t *output_strides,
    const ptrdiff_t *input_standardization_strides,
    const ptrdiff_t *input_std_deviation_strides,
    const ptrdiff_t *input_strides,
    ptrdiff_t weight_stride,
    ptrdiff_t bias_stride,
    bool bias_exist) {
    size_t b0 = blockIdx.x, b1 = blockIdx.y;

    auto output_ptr = output + b0 * output_strides[0] + b1 * output_strides[1];
    auto input_ptr = input + b0 * input_strides[0] + b1 * input_strides[1];
    auto standard_ptr = input_standardization + b0 * input_standardization_strides[0] + b1 * input_standardization_strides[1];
    auto std_ptr = input_std_deviation + b0 * input_std_deviation_strides[0] + b1 * input_std_deviation_strides[1];
    Tcompute mean = op::common_cuda::reduce_op::sum<BLOCK_SIZE, Tdata, Tcompute>(
                        input_ptr,
                        normalized_size)
                  / normalized_size;
    Tcompute sum_squared = op::common_cuda::reduce_op::sumSquared<BLOCK_SIZE, Tdata, Tcompute>(
        input_ptr,
        normalized_size);

    Tcompute var = sum_squared / normalized_size - mean * mean;
    Tcompute std_deviation = sqrtf(var + Tcompute(eps));
    *std_ptr = std_deviation;

    for (size_t d = 0; d < normalized_size; d++) {
        Tcompute x_standard = (Tcompute(input_ptr[d]) - mean) / std_deviation;
        standard_ptr[d] = x_standard;
        output_ptr[d] = x_standard * Tcompute(*(weight + d * weight_stride)) + (bias_exist ? Tcompute(*(bias + d * bias_stride)) : Tcompute(0));
    }
}

template <typename T, int BLOCK_SIZE>
__device__ void blockLayernormKernel(T *output, T const *input, T const *weight, T const *bias, float eps, int dimsize,
                                     const ptrdiff_t *output_strides,
                                     const ptrdiff_t *input_strides,
                                     const size_t *shape,
                                     ptrdiff_t weight_stride,
                                     ptrdiff_t bias_stride,
                                     int ndim,
                                     bool bias_exist) {
    // 只能处理axis=-1
    int ind_i = 0; // input id
    int ind_o = 0; // output id
    int tid = blockIdx.x;
    for (int j = ndim - 2; j >= 0; j--) {
        ind_i += (tid % (int)shape[j]) * (int)input_strides[j];
        ind_o += (tid % (int)shape[j]) * (int)output_strides[j];
        tid = tid / (int)shape[j];
    }

    float mu_partial = op::common_cuda::reduce_op::sum<BLOCK_SIZE, T, float>(
                           input + ind_i,
                           dimsize)
                     / dimsize;
    __shared__ float mu;
    if (threadIdx.x == 0) {
        mu = mu_partial;
    } // threadIdx.x = 0对应的是全局sum
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    __syncthreads();
    float sigma2_partial = 0.0f;
    for (int id = threadIdx.x; id < dimsize; id += BLOCK_SIZE) {
        sigma2_partial += (static_cast<float>(input[ind_i + id]) - mu) * (static_cast<float>(input[ind_i + id]) - mu);
    }

    __shared__ float sigma2;
    float sigma2_block = BlockReduce(temp_storage).Sum(sigma2_partial);
    if (threadIdx.x == 0) {
        float sigma_tmp = sqrt(sigma2_block * __fdividef(1.0F, dimsize) + eps);
        sigma2 = __fdividef(1.0F, sigma_tmp);
    }
    __syncthreads();
    for (int id = threadIdx.x; id < dimsize; id += BLOCK_SIZE) {
        output[ind_o + id] = static_cast<T>(static_cast<float>(weight[id * weight_stride]) * (static_cast<float>(input[ind_i + id]) - mu) * sigma2 + (bias_exist ? static_cast<float>(bias[id * bias_stride]) : 0.0f));
    }
}
template <typename T>
struct SumOp {
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return a + b;
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
template <typename T, int BLOCK_SIZE_x, int BLOCK_SIZE_y>
__device__ void warpLayernormKernel(T *output, T const *input, T const *weight, T const *bias, float eps, int othersize, int dimsize,
                                    const ptrdiff_t *output_strides,
                                    const ptrdiff_t *input_strides,
                                    const size_t *shape,
                                    ptrdiff_t weight_stride,
                                    ptrdiff_t bias_stride,
                                    int ndim,
                                    bool bias_exist) {
    // 默认dimsize < 1024
    int ind_i = 0; // input id
    int ind_o = 0; // output id
    int tid = blockIdx.x * blockDim.y + threadIdx.y;
    if (tid < othersize) {
        for (int j = ndim - 2; j >= 0; j--) {
            ind_i += (tid % (int)shape[j]) * (int)input_strides[j];
            ind_o += (tid % (int)shape[j]) * (int)output_strides[j];
            tid = tid / (int)shape[j];
        }

        float mu_partial = 0.0f;
        for (int id = threadIdx.x; id < dimsize; id += BLOCK_SIZE_x) {
            mu_partial += static_cast<float>(input[ind_i + id]);
        }
        mu_partial = WarpAllReduce<SumOp, float, BLOCK_SIZE_x>(mu_partial);
        __shared__ float mu[BLOCK_SIZE_y];

        if (threadIdx.x == 0) {
            mu[threadIdx.y] = mu_partial * __fdividef(1.0F, dimsize);
        } // threadIdx.x = 0对应的是全局sum
        __syncthreads();
        float sigma2_partial = 0.0f;
        for (int id = threadIdx.x; id < dimsize; id += BLOCK_SIZE_x) {
            sigma2_partial += (static_cast<float>(input[ind_i + id]) - mu[threadIdx.y]) * (static_cast<float>(input[ind_i + id]) - mu[threadIdx.y]);
        }
        sigma2_partial = WarpAllReduce<SumOp, float, BLOCK_SIZE_x>(sigma2_partial);
        __shared__ float sigma2[BLOCK_SIZE_y];

        if (threadIdx.x == 0) {
            float sigma_tmp = sqrt(sigma2_partial * __fdividef(1.0F, dimsize) + eps);
            sigma2[threadIdx.y] = __fdividef(1.0F, sigma_tmp);
        }
        __syncthreads();
        for (int id = threadIdx.x; id < dimsize; id += BLOCK_SIZE_x) {
            output[ind_o + id] = static_cast<T>(static_cast<float>(weight[id * weight_stride]) * (static_cast<float>(input[ind_i + id]) - mu[threadIdx.y]) * sigma2[threadIdx.y] + (bias_exist ? static_cast<float>(bias[id * bias_stride]) : 0.0f));
        }
    }
}
#endif // __LAYER_NORM_KERNEL_CUH__
