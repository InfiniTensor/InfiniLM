#ifndef __VAR_MEAN_CUDA_H__
#define __VAR_MEAN_CUDA_H__

#include <cmath> // NAN

namespace device {
namespace cuda {
template <typename Tdata>
__inline__ __device__ Tdata Nan();
template <>
__inline__ __device__ float Nan<float>() {
    return NAN;
}
template <>
__inline__ __device__ double Nan<double>() {
    return NAN;
}
template <>
__inline__ __device__ half Nan<half>() {
    return __float2half(NAN);
}

#if defined(ENABLE_MOORE_API)
using bf16_t = __mt_bfloat16;
#elif defined(ENABLE_METAX_API)
using bf16_t = __hpcc_bfloat16;
#else
using bf16_t = __nv_bfloat16;
#endif

/* bf16 */
template <>
__inline__ __device__ bf16_t Nan<bf16_t>() {
    return __float2bfloat16_rn(NAN);
}

template <typename Tdata>
__inline__ __device__ Tdata Div(Tdata a, Tdata b);
template <>
__inline__ __device__ float Div<float>(float a, float b) {
#ifdef OF_LAYER_NORM_USE_FAST_MATH
    return __fdividef(a, b);
#else
    return a / b;
#endif
}
template <>
__inline__ __device__ double Div<double>(double a, double b) {
    return a / b;
}
template <>
__inline__ __device__ half Div<half>(half a, half b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return __hdiv(a, b);
#else
    return __float2half(__half2float(a) / __half2float(b));
#endif
}
template <>
__inline__ __device__ bf16_t Div<bf16_t>(bf16_t a, bf16_t b) {

#if defined(ENABLE_NVIDIA_API) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return __hdiv(a, b);
#else
    return __float2bfloat16_rn(
        __bfloat162float(a) / __bfloat162float(b));
#endif
}

template <typename Tdata, typename ComputeType>
inline __device__ void WelfordReduce(const Tdata *input_ptr, ComputeType &mean, ComputeType &m2, ComputeType &count,
                                     const size_t start, const size_t end, const size_t step,
                                     const size_t ndim, const size_t *shape, const ptrdiff_t *strides) {
    ComputeType old_mean = 0.0;
    for (size_t i = start; i < end; i += step) {
        ++count;
        old_mean = mean;
        size_t input_offset = indexToOffset(i, ndim, shape, strides);
        ComputeType input_value = static_cast<ComputeType>(input_ptr[input_offset]);
        mean += (input_value - mean) / count;
        m2 += (input_value - mean)
            * (input_value - old_mean);
    }
}

template <typename Tdata>
inline __device__ void WelfordCombine(Tdata val, Tdata &mean, Tdata &m2, Tdata &count) {
    count += 1;
    Tdata delta1 = val - mean;
    mean += Div(delta1, count);
    Tdata delta2 = val - mean;
    m2 += delta1 * delta2;
}

template <typename Tdata>
inline __device__ void WelfordCombine(Tdata b_mean, Tdata b_m2, Tdata b_count, Tdata &mean, Tdata &m2, Tdata &count) {
    if (b_count == 0) {
        return;
    }
    Tdata new_count = count + b_count;              // n1 + n2
    Tdata nb_over_n = Div(b_count, new_count);      // n2 / (n1 + n2)
    Tdata delta = b_mean - mean;                    // mean2 - mean1
    mean += delta * nb_over_n;                      // mean1 + n2 * (mean2 - mean1) / (n1 + n2)
    m2 += b_m2 + delta * delta * count * nb_over_n; // m21 + m22 + n2 * (mean2 - mean1) ^ 2 / (n1 + n2)
    count = new_count;
}

template <typename Tdata>
inline __device__ void WelfordCombineLoop(const Tdata *b_mean, const Tdata *b_m2, const Tdata *b_count,
                                          Tdata &mean, Tdata &m2, Tdata &count,
                                          const size_t start, const size_t end, const size_t step) {
    for (size_t i = start; i < end; i += step) {
        WelfordCombine(b_mean[i], b_m2[i], b_count[i], mean, m2, count);
    }
}

template <typename Tdata, int thread_group_width = 32>
__inline__ __device__ void WelfordWarpReduce(Tdata thread_mean, Tdata thread_m2, Tdata thread_count,
                                             Tdata &mean, Tdata &m2, Tdata &count) {
    mean = thread_mean;
    m2 = thread_m2;
    count = thread_count;
    for (int lane_mask = thread_group_width / 2; lane_mask > 0; lane_mask /= 2) {
        Tdata b_mean = __shfl_down_sync(0xffffffff, mean, lane_mask, thread_group_width);
        Tdata b_m2 = __shfl_down_sync(0xffffffff, m2, lane_mask, thread_group_width);
        Tdata b_count = __shfl_down_sync(0xffffffff, count, lane_mask, thread_group_width);
        WelfordCombine(b_mean, b_m2, b_count, mean, m2, count);
    }
}

template <typename Tdata, size_t kWarpSize = 32>
__inline__ __device__ void WelfordBlockAllReduce(Tdata thread_mean, Tdata thread_m2, Tdata thread_count,
                                                 Tdata &result_mean, Tdata &result_m2, Tdata &result_count) {
    __shared__ Tdata mean_shared[kWarpSize];
    __shared__ Tdata m2_shared[kWarpSize];
    __shared__ Tdata count_shared[kWarpSize];
    __shared__ Tdata mean_result_broadcast;
    __shared__ Tdata m2_result_broadcast;
    __shared__ Tdata count_result_broadcast;
    const int lid = threadIdx.x % kWarpSize;
    const int wid = threadIdx.x / kWarpSize;
    // warp内规约
    Tdata warp_mean = 0.0;
    Tdata warp_m2 = 0.0;
    Tdata warp_count = 0;
    WelfordWarpReduce(thread_mean, thread_m2, thread_count, warp_mean, warp_m2, warp_count);
    __syncthreads();
    if (lid == 0) { // 每个warp内的的thread0 保存warp结果
        mean_shared[wid] = warp_mean;
        m2_shared[wid] = warp_m2;
        count_shared[wid] = warp_count;
    }
    __syncthreads();
    // warp间规约
    if (wid == 0) {
        if (threadIdx.x < blockDim.x / kWarpSize) {
            warp_mean = mean_shared[lid];
            warp_m2 = m2_shared[lid];
            warp_count = count_shared[lid];
        } else {
            warp_mean = static_cast<Tdata>(0);
            warp_m2 = static_cast<Tdata>(0);
            warp_count = static_cast<Tdata>(0);
        }
        __syncwarp();
        Tdata block_mean = 0;
        Tdata block_m2 = 0;
        Tdata block_count = 0;
        WelfordWarpReduce(warp_mean, warp_m2, warp_count, block_mean, block_m2, block_count);
        if (lid == 0) {
            mean_result_broadcast = block_mean;
            m2_result_broadcast = block_m2;
            count_result_broadcast = block_count;
        }
    }
    __syncthreads();
    result_mean = mean_result_broadcast;
    result_m2 = m2_result_broadcast;
    result_count = count_result_broadcast;
}
} // namespace cuda
} // namespace device

__device__ int32_t done_block_count = 0;

template <typename Tdata, typename ComputeType>
__global__ void ComputeVarScalarOut(const Tdata *input_ptr, Tdata *var_output_ptr, Tdata *mean_output_ptr, ComputeType *tmp_buffer_ptr,
                                    size_t input_size, size_t input_ndim, size_t *permuted_input_shape, ptrdiff_t *permuted_input_strides,
                                    bool unbiased, bool is_nan) {
    // 处理 NaN 情况
    if (is_nan) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            *var_output_ptr = device::cuda::Nan<Tdata>();
            mean_output_ptr[0] = (input_size == 0) ? device::cuda::Nan<Tdata>() : input_ptr[0];
        }
        return;
    }

    // 计算每个 block 和 thread 的工作量
    const size_t elems_per_block = input_size / gridDim.x;
    const size_t elems_per_thread = elems_per_block / blockDim.x;
    // 线程级 Welford 累积
    ComputeType thread_mean = 0.0, thread_m2 = 0.0, thread_count = 0;

    // 每个线程处理常规元素（stride 访问）
    if (elems_per_thread > 0) {
        const size_t block_start = blockIdx.x * elems_per_block;
        const size_t regular_elems = elems_per_block - (elems_per_block % blockDim.x);
        device::cuda::WelfordReduce<Tdata, ComputeType>(input_ptr, thread_mean, thread_m2, thread_count,
                                                        /*start=*/block_start + threadIdx.x, /*end=*/block_start + regular_elems, /*step=*/blockDim.x,
                                                        /*ndim=*/input_ndim, /*shape=*/permuted_input_shape, /*strides=*/permuted_input_strides);
    }

    // thread 0 处理本 block 的尾部元素以及跨 block 的尾部元素（单个线程处理）
    if (threadIdx.x == 0) {
        size_t tail_count = elems_per_block % blockDim.x;
        // 最后一个 block 还需要处理总元素数的尾部
        if (blockIdx.x == gridDim.x - 1) {
            tail_count += input_size % gridDim.x;
        }
        if (tail_count > 0) {
            const size_t tail_start = blockIdx.x * elems_per_block + blockDim.x * elems_per_thread;
            device::cuda::WelfordReduce<Tdata, ComputeType>(input_ptr, thread_mean, thread_m2, thread_count,
                                                            /*start=*/tail_start, /*end=*/tail_start + tail_count, /*step=*/1,
                                                            /*ndim=*/input_ndim, /*shape=*/permuted_input_shape, /*strides=*/permuted_input_strides);
        }
    }

    // Block 级规约
    ComputeType block_mean = 0.0, block_m2 = 0.0, block_count = 0;
    device::cuda::WelfordBlockAllReduce<ComputeType>(thread_mean, thread_m2, thread_count,
                                                     block_mean, block_m2, block_count);

    // 单 block 情况：直接输出结果
    if (gridDim.x == 1) {
        if (threadIdx.x == 0) {
            ComputeType divisor = unbiased ? block_count - 1 : block_count;
            var_output_ptr[0] = device::cuda::Div(block_m2, divisor);
            mean_output_ptr[0] = static_cast<Tdata>(block_mean);
        }
        return;
    }

    // 多 block 情况：使用临时缓冲区
    ComputeType *tmp_mean_ptr = tmp_buffer_ptr;
    ComputeType *tmp_m2_ptr = tmp_mean_ptr + gridDim.x;
    ComputeType *tmp_count_ptr = tmp_m2_ptr + gridDim.x;

    // 保存本 block 的结果
    if (threadIdx.x == 0) {
        tmp_mean_ptr[blockIdx.x] = block_mean;
        tmp_m2_ptr[blockIdx.x] = block_m2;
        tmp_count_ptr[blockIdx.x] = block_count;
    }

    // 最后一个 block 负责最终规约
    __shared__ bool is_last_block;
    if (threadIdx.x == 0) {
        is_last_block = (atomicAdd(&done_block_count, 1) == gridDim.x - 1);
    }
    __syncthreads();

    if (is_last_block) {
        // 每个线程合并一部分 block 的结果
        ComputeType final_thread_mean = 0.0, final_thread_m2 = 0.0, final_thread_count = 0;
        const size_t blocks_per_thread = gridDim.x / blockDim.x;
        const size_t regular_blocks = blocks_per_thread * blockDim.x;

        if (blocks_per_thread > 0) {
            device::cuda::WelfordCombineLoop(tmp_mean_ptr, tmp_m2_ptr, tmp_count_ptr,
                                             final_thread_mean, final_thread_m2, final_thread_count,
                                             /*start=*/threadIdx.x, /*end=*/regular_blocks, /*step=*/blockDim.x);
        }

        // thread 0 处理尾部 block
        if (threadIdx.x == 0 && regular_blocks < gridDim.x) {
            device::cuda::WelfordCombineLoop(&tmp_mean_ptr[regular_blocks], &tmp_m2_ptr[regular_blocks], &tmp_count_ptr[regular_blocks],
                                             final_thread_mean, final_thread_m2, final_thread_count,
                                             /*start=*/0, /*end=*/gridDim.x - regular_blocks, /*step=*/1);
        }

        // 最终 block 级规约并输出
        ComputeType final_mean = 0, final_m2 = 0, final_count = 0;
        device::cuda::WelfordBlockAllReduce<ComputeType>(final_thread_mean, final_thread_m2, final_thread_count,
                                                         final_mean, final_m2, final_count);
        if (threadIdx.x == 0) {
            ComputeType divisor = unbiased ? final_count - 1 : final_count;
            var_output_ptr[0] = device::cuda::Div(final_m2, divisor);
            mean_output_ptr[0] = static_cast<Tdata>(final_mean);
            done_block_count = 0; // 重置计数器
        }
    }
}

// CUDA: grid stride looping
#define CUDA_1D_KERNEL_LOOP(i, n)                                                                  \
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x, step = blockDim.x * gridDim.x; i < (n); \
         i += step)

template <typename Tdata, typename ComputeType>
__forceinline__ __device__ __host__ void ComputeVarMeanUsingWelford(
    const Tdata *input_ptr,
    size_t offset,
    Tdata &var_output,
    Tdata &mean_output,
    size_t reduce_num,
    size_t input_ndim,
    size_t *permuted_input_shape,
    ptrdiff_t *permuted_input_strides,
    bool unbiased) {
    size_t count = 0;
    ComputeType mean = 0.0;
    ComputeType old_mean = 0.0;
    ComputeType m2 = 0.0;
    for (size_t i = 0; i < reduce_num; ++i) {
        size_t input_offset = indexToOffset(offset + i, input_ndim, permuted_input_shape, permuted_input_strides);
        count++;
        old_mean = mean;
        mean = old_mean + (static_cast<ComputeType>(input_ptr[input_offset]) - old_mean) / count;
        m2 += (static_cast<ComputeType>(input_ptr[input_offset]) - old_mean) * (static_cast<ComputeType>(input_ptr[input_offset]) - mean);
    }
    var_output = static_cast<Tdata>(m2 / (unbiased ? count - 1 : count));
    mean_output = static_cast<Tdata>(mean);
}

template <typename Tdata, typename ComputeType>
__global__ void ComputeVarMeanUsingWelfordWrapper(
    const Tdata *input_ptr, Tdata *var_output_ptr, Tdata *mean_output_ptr,
    size_t input_ndim,
    size_t output_size,
    size_t reduce_num,
    size_t *permuted_input_shape,
    ptrdiff_t *permuted_input_strides,
    bool unbiased,
    bool is_nan) {
    if (is_nan) {
        if (reduce_num == 0) {
            CUDA_1D_KERNEL_LOOP(i, output_size) {
                var_output_ptr[i] = device::cuda::Nan<Tdata>();
                mean_output_ptr[i] = device::cuda::Nan<Tdata>();
            }
        } else {
            CUDA_1D_KERNEL_LOOP(i, output_size) {
                const size_t input_offset = indexToOffset(i * reduce_num, input_ndim, permuted_input_shape, permuted_input_strides);
                var_output_ptr[i] = device::cuda::Nan<Tdata>();
                mean_output_ptr[i] = input_ptr[input_offset];
            }
        }
    } else {
        CUDA_1D_KERNEL_LOOP(i, output_size) {
            ComputeVarMeanUsingWelford<Tdata, ComputeType>(
                input_ptr,
                i * reduce_num,
                var_output_ptr[i],
                mean_output_ptr[i],
                reduce_num,
                input_ndim,
                permuted_input_shape,
                permuted_input_strides,
                unbiased);
        }
    }
}

#endif // __VAR_MEAN_CUDA_H__
