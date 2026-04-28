#ifndef __INFINIOP_COMMON_CPU_H__
#define __INFINIOP_COMMON_CPU_H__

#include "../../../utils.h"
#include "cpu_handle.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#ifdef ENABLE_OMP
#include <omp.h>
#endif

namespace op::common_cpu {

// return the memory offset a tensor given flattened index
size_t indexToOffset(size_t flat_index, size_t ndim, const size_t *shape, const ptrdiff_t *strides);

/**
 * get the total array size (element count) after applying padding for a
 * ndim-ary tensor with the given shape
 */
size_t getPaddedSize(size_t ndim, size_t *shape, const size_t *pads);

// calculate the padded shape and store the result in padded_shape
std::vector<size_t> getPaddedShape(size_t ndim, const size_t *shape, const size_t *pads);

} // namespace op::common_cpu

#endif // __INFINIOP__COMMON_CPU_H__
