#ifndef _TOPKSOFTMAX_KERNEL_CUH__
#define _TOPKSOFTMAX_KERNEL_CUH__
#include <cfloat>
#include <cub/block/block_load.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_store.cuh>

template <typename T>
inline __device__ float exp_func(T x) {
    float data;
    if constexpr (std::is_same_v<T, float>) {
        data = x;
    } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
        data = __bfloat162float(x);
    } else if constexpr (std::is_same_v<T, half>) {
        data = __half2float(x);
    }
    return __expf(data);
}

// Warp-level sum reduction for Hygon platform
template <int warp_threads>
__inline__ __device__ float WarpSum(float val) {
    for (int mask = warp_threads / 2; mask > 0; mask /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template <typename T, int BLOCK_SIZE = 128>
__global__ void softmax_topk_row_kernel(float *values_topk, // 输出数据, 形状[N, topk]
                                        int *indices_topk,  // 输出索引, 形状[N, topk]
                                        const T *input,     // 输入数据 [N, width]
                                        const size_t N,
                                        const size_t width,
                                        const size_t topk,
                                        bool norm

) {
    const int bid = blockIdx.x;
    if (bid >= N) {
        return;
    }

    const int tid = threadIdx.x;
    const T *data_input = input + bid * width;
    float *values_topk_output = values_topk + bid * topk;
    int *indices_topk_output = indices_topk + bid * topk;

    const int warp_id = tid / 32;

    __shared__ T shared_max;
    __shared__ float shared_sum;
    typedef cub::BlockReduce<T, BLOCK_SIZE> BlockReduce;

    // ------------------------------------------------ //
    //             第一步：计算最大值                      //
    // ------------------------------------------------ //
    T thread_max = data_input[0];
    if (tid < width) {
        thread_max = thread_max > data_input[tid] ? thread_max : data_input[tid];
    }

    {
        __shared__ typename BlockReduce::TempStorage temp_storage_max;
#if CUDART_VERSION >= 12090
        T value_max = BlockReduce(temp_storage_max).Reduce(thread_max, ::cuda::maximum());
#elif defined(ENABLE_HYGON_API)
        T value_max = BlockReduce(temp_storage_max).Reduce(
            thread_max, [](const T &a, const T &b) { return (a > b) ? a : b; }, BLOCK_SIZE);
#else
        T value_max = BlockReduce(temp_storage_max).Reduce(thread_max, cub::Max());
#endif
        if (tid == 0) {
            shared_max = value_max;
        }
    }
    __syncthreads();

    // ------------------------------------------------ //
    //             第二步：计算指数和                      //
    // ------------------------------------------------ //
    float exp_val = 0.0f;
    if (tid < width) {
        T temp_val = data_input[tid] - shared_max;
        exp_val = exp_func<T>(temp_val);
    }

    {
        __shared__ typename BlockReduce::TempStorage temp_storage_sum;
        float value_sum = BlockReduce(temp_storage_sum).Sum(exp_val);
        if (tid == 0) {
            shared_sum = value_sum;
        }
    }
    __syncthreads();

    // ------------------------------------------------ //
    //           第三步：计算 Softmax                     //
    // ------------------------------------------------ //
    exp_val /= shared_sum;

    // ------------------------------------------------ //
    //           第四步：计算 排序                        //
    // ------------------------------------------------ //
    float thread_values[1] = {-FLT_MAX};
    int thread_indices[1] = {-1};
    if ((tid < width) && (exp_val > thread_values[0])) {
        thread_values[0] = exp_val;
        thread_indices[0] = tid;
    }
    {
        typedef cub::BlockRadixSort<float, BLOCK_SIZE, 1, int> BlockRadixSort;
        __shared__ typename BlockRadixSort::TempStorage temp_storage;
        BlockRadixSort(temp_storage).SortDescending(thread_values, thread_indices);
    }
    __syncthreads();

    if (0 == warp_id) {
        int indice = -1;
        float value = 0.0f;
        if (tid < topk) {
            indice = thread_indices[0];
            value = thread_values[0];
        }

        // ------------------------------------------------ //
        //           第五步： topk的和                         //
        // ------------------------------------------------ //
        {
#ifdef ENABLE_HYGON_API
            float warp_sum = WarpSum<32>(value);
            if (0 == tid) {
                shared_sum = warp_sum + 1e-9f;
            }
#else
            typedef cub::WarpReduce<float, 32> WarpReduce;
            __shared__ typename WarpReduce::TempStorage temp_storage;
            float warp_sum = WarpReduce(temp_storage).Sum(value);
            if (0 == tid) {
                shared_sum = warp_sum + 1e-9f;
            }
#endif
        }
        __syncwarp();

        // ------------------------------------------------ //
        //           第6步： norm归一化                       //
        // ------------------------------------------------ //
        if (norm && (tid < topk)) {
            value /= shared_sum;
        }

        // ------------------------------------------------ //
        //           第7步： 最终的返回值                       //
        // ------------------------------------------------ //
        if (tid < topk) {
            values_topk_output[tid] = value;
            indices_topk_output[tid] = indice;
        }
    }
}

#endif // _TOPKSOFTMAX_KERNEL_CUH__
