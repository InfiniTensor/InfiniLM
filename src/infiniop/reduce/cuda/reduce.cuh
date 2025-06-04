#ifndef __INFINIOP_REDUCE_CUDA_H__
#define __INFINIOP_REDUCE_CUDA_H__

#include <cub/block/block_reduce.cuh>

/*
 * Device functions for reduction operations on CUDA.
 *
 * Note: Only local result on thread 0 is guranteed to be correct.
 *       A manual broadcast is needed for other threads.
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
        max_ = cub::Max()(max_, data_ptr[i]);
    }

    using BlockReduce = cub::BlockReduce<Tdata, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    return BlockReduce(temp_storage).Reduce(max_, cub::Max(), BLOCK_SIZE);
}

} // namespace op::common_cuda::reduce_op

#endif
