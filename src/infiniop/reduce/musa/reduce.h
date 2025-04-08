#ifndef __INFINIOP_REDUCE_MUSA_H__
#define __INFINIOP_REDUCE_MUSA_H__

#include <cub/block/block_reduce.cuh>

namespace op::common_musa::reduce_op {

template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
__device__ __forceinline__ Tcompute sumSquared(const Tdata *data_ptr, size_t count) {
    Tcompute ss = 0;

    // Each thread computes its partial sum
    for (size_t i = threadIdx.x; i < count; i += BLOCK_SIZE) {
        ss += Tcompute(data_ptr[i] * data_ptr[i]);
    }

    // Use CUB block-level reduction
    using BlockReduce = cub::BlockReduce<Tcompute, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    return BlockReduce(temp_storage).Sum(ss);
}

} // namespace op::common_musa::reduce_op

#endif
