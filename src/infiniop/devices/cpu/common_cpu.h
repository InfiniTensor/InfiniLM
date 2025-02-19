#ifndef __INFINIOP__COMMON_CPU_H__
#define __INFINIOP__COMMON_CPU_H__

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

// convert half-precision float to single-precision float
float f16_to_f32(uint16_t code);

// convert single-precision float to half-precision float
uint16_t f32_to_f16(float val);

// return the memory offset of original tensor, given the flattened index of broadcasted tensor
size_t indexToReducedOffset(size_t flat_index, size_t ndim, ptrdiff_t const *broadcasted_strides, ptrdiff_t const *target_strides);

// return the memory offset a tensor given flattened index
size_t indexToOffset(size_t flat_index, size_t ndim, size_t const *shape, ptrdiff_t const *strides);

/**
 * get the total array size (element count) after applying padding for a
 * ndim-ary tensor with the given shape
 */
size_t getPaddedSize(size_t ndim, size_t *shape, size_t const *pads);

// calculate the padded shape and store the result in padded_shape
std::vector<size_t> getPaddedShape(size_t ndim, size_t const *shape, size_t const *pads);

#endif // __INFINIOP__COMMON_CPU_H__
