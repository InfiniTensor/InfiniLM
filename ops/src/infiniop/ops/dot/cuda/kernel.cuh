#pragma once

#include "../../../reduce/cuda/reduce.cuh"
#include <type_traits>

namespace op::cuda {

template <unsigned int BLOCK_SIZE, typename Tdata, typename Tout, typename Tcompute>
__global__ void dot_kernel(
    Tout *result,
    const Tdata *a,
    const Tdata *b,
    size_t n,
    ptrdiff_t a_stride,
    ptrdiff_t b_stride) {

    Tcompute sum = 0;

    for (size_t i = threadIdx.x; i < n; i += BLOCK_SIZE) {
        sum += Tcompute(a[i * a_stride]) * Tcompute(b[i * b_stride]);
    }

    using BlockReduce = cub::BlockReduce<Tcompute, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    Tcompute block_sum = BlockReduce(temp_storage).Sum(sum);

    if (threadIdx.x == 0) {
        result[0] = static_cast<Tout>(block_sum);
    }
}

} // namespace op::cuda
