#ifndef _TOPKROUTER_KERNEL_CUH__
#define _TOPKROUTER_KERNEL_CUH__

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

template <typename T>
inline __device__ T sigmoid_func(T x) {
    // sigmoid(x) = 1 / (1 + exp(-x))
    return 1.0f / (1.0f + exp_func<T>(-x));
}

struct CustomLess {
    template <typename DataType>
    __device__ bool operator()(const DataType &lhs, const DataType &rhs) {
        return lhs > rhs;
    }
};

// Warp-level sum reduction for Hygon platform
template <int warp_threads>
__inline__ __device__ float WarpSum(float val) {
    for (int mask = warp_threads / 2; mask > 0; mask /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template <typename T, int BLOCK_THREADS = 256>
__global__ void topkrouter_kernel(float *values_topk,             // 输出数据, 形状[N, topk]
                                  int *indices_topk,              // 输出索引, 形状[N, topk]
                                  const T *input,                 // 输入数据 [N, width]
                                  const float *d_correction_bias, // 输入数据 [width]
                                  const float routed_scaling_factor,
                                  const size_t N,
                                  const size_t width,
                                  const size_t topk

) {
    const int bid = blockIdx.x;
    if (bid >= N) {
        return;
    }
    const int tid = threadIdx.x;
    const T *data_input = input + bid * width;
    float *values_topk_output = values_topk + bid * topk;
    int *indices_topk_output = indices_topk + bid * topk;

    constexpr int warp_threads = 32;
    constexpr int block_threads = 256;
    constexpr int warps_per_block = block_threads / warp_threads;
    const int warp_id = tid / warp_threads;
    const int lane_id = tid % warp_threads;

    __shared__ float share_data[256];
    __shared__ float share_data_group[8];
    __shared__ float share_data_group_mask[8]; // 有效的group
    __shared__ float share_sum;
    if (tid < 8) {
        share_data_group_mask[tid] = 0.0f;
    }

    // ------------------------------------------------------ //
    //             对输入数据做 sigmoid                         //
    // ------------------------------------------------------ //
    float value = sigmoid_func(data_input[tid]);

    // ------------------------------------------------------ //
    //             对输入数据加偏执                              //
    // ------------------------------------------------------ //
    value += d_correction_bias[tid];

    // ----------------------------------------------------------- //
    //      每个warp为一组，一共8组，找出每组的最大的前两个数据            //
    // ----------------------------------------------------------- //
    float thread_values[1] = {value};
    int thread_indices[1] = {tid};
    using WarpMergeSortT = cub::WarpMergeSort<float, 1, warp_threads, int>;
    {
        __shared__ typename WarpMergeSortT::TempStorage temp_storage[warps_per_block];
        WarpMergeSortT(temp_storage[warp_id]).Sort(thread_values, thread_indices, CustomLess());
    }
    __syncthreads();
    share_data[tid] = thread_values[0];

    // ----------------------------------------------------------- //
    //              每个组中,前两个数据的和                            //
    // ----------------------------------------------------------- //
    __syncthreads();
    if (0 == lane_id) {
        share_data_group[warp_id] = share_data[warp_id * warp_threads] + share_data[warp_id * warp_threads + 1];
    }
    __syncthreads();
    // ----------------------------------------------------------- //
    //                  再选前 4 个                                 //
    // ----------------------------------------------------------- //
    if (0 == warp_id) {
        float thread_values[1] = {-FLT_MAX};
        int thread_indices[1] = {-1};
        if (lane_id < 8) {
            thread_values[0] = share_data_group[lane_id];
            thread_indices[0] = lane_id;
        }

        {
            __shared__ typename WarpMergeSortT::TempStorage temp_storage[1];
            WarpMergeSortT(temp_storage[0]).Sort(thread_values, thread_indices, CustomLess());
        }
        if (lane_id < 4) {
            int indices = thread_indices[0];
            share_data_group_mask[indices] = 1.0f;
        }
    }
    __syncthreads();

    // ----------------------------------------------------------- //
    //                 求得 最后一次topk                             //
    // ----------------------------------------------------------- //
    value = value * share_data_group_mask[warp_id];
    thread_values[0] = value;
    thread_indices[0] = tid;
    {
        typedef cub::BlockRadixSort<float, BLOCK_THREADS, 1, int> BlockRadixSort;
        __shared__ typename BlockRadixSort::TempStorage temp_storage;
        BlockRadixSort(temp_storage).SortDescending(thread_values, thread_indices);
    }
    __syncthreads();

    // ----------------------------------------------------------- //
    //                 归一化                                       //
    // ----------------------------------------------------------- //
    if (0 == warp_id) {
        value = 0.0f;
        if (tid < 8) {
            int index = thread_indices[0];
            value = sigmoid_func(data_input[index]);
        }
        {
#ifdef ENABLE_HYGON_API
            float warp_sum = WarpSum<warp_threads>(value);
            if (0 == tid) {
                share_sum = warp_sum + 1e-9f;
            }
#else
            typedef cub::WarpReduce<float, warp_threads> WarpReduce;
            __shared__ typename WarpReduce::TempStorage temp_storage;
            float warp_sum = WarpReduce(temp_storage).Sum(value);
            if (0 == tid) {
                share_sum = warp_sum + 1e-9f;
            }
#endif
        }
        __syncwarp();

        if (tid < 8) {
            int index = thread_indices[0];
            indices_topk_output[tid] = index;
            values_topk_output[tid] = routed_scaling_factor * value / share_sum;
        }
    }
}

#endif // _topkrouter_KERNEL_CUH__
