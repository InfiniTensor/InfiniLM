#ifndef __INFINIOP_REDUCE_CUDA_H__
#define __INFINIOP_REDUCE_CUDA_H__
#include <cub/block/block_reduce.cuh>
/*
 * Device functions for reduction operations on CUDA.
 *
 * Note: Only local result on thread 0 is guranteed to be correct.
 *       A manual broadcast is needed for other threads.
 *
 * Important Note: This is a device-independent header file containing reduce kernels
 *                 for all cuda-supporting platforms. Include device-specific headers
 *                 (such as <cub/block/block_reduce.cuh> for nvidia) in your source file
 *                 and then include this file for proper usage.
 */
namespace op::common_cuda::reduce_op {

// Sum(x^2) on contiguous data of length count
template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
__device__ __forceinline__ Tcompute sumSquared(const Tdata *data_ptr, size_t count) {
    Tcompute ss = 0;

    // Each thread computes its partial sum
    for (size_t i = threadIdx.x; i < count; i += BLOCK_SIZE) {
        ss += Tcompute(data_ptr[i]) * Tcompute(data_ptr[i]);
    }

    // Use CUB block-level reduction
    using BlockReduce = cub::BlockReduce<Tcompute, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    return BlockReduce(temp_storage).Sum(ss);
}

// Sum(x) on contiguous data of length count
template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
__device__ __forceinline__ Tcompute sum(const Tdata *data_ptr, size_t count) {
    Tcompute s = 0;

    for (size_t i = threadIdx.x; i < count; i += BLOCK_SIZE) {
        s += Tcompute(data_ptr[i]);
    }

    using BlockReduce = cub::BlockReduce<Tcompute, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    return BlockReduce(temp_storage).Sum(s);
}

// Max(x) on contiguous data of length count
template <unsigned int BLOCK_SIZE, typename Tdata>
__device__ __forceinline__ Tdata max(const Tdata *data_ptr, size_t count) {
    Tdata max_ = data_ptr[0];

    for (size_t i = threadIdx.x; i < count; i += BLOCK_SIZE) {
#ifdef ENABLE_HYGON_API
        max_ = (data_ptr[i] > max_) ? data_ptr[i] : max_;
#else
#if CUDART_VERSION >= 12090
        max_ = ::cuda::maximum()(max_, data_ptr[i]);
#else
        max_ = cub::Max()(max_, data_ptr[i]);
#endif
#endif
    }

    using BlockReduce = cub::BlockReduce<Tdata, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

#ifdef ENABLE_HYGON_API
    return BlockReduce(temp_storage).Reduce(
        max_, [](const Tdata &a, const Tdata &b) { return (a > b) ? a : b; }, BLOCK_SIZE);
#else
#if CUDART_VERSION >= 12090
    return BlockReduce(temp_storage).Reduce(max_, ::cuda::maximum(), BLOCK_SIZE);
#else
    return BlockReduce(temp_storage).Reduce(max_, cub::Max(), BLOCK_SIZE);
#endif
#endif
}

} // namespace op::common_cuda::reduce_op

#endif
