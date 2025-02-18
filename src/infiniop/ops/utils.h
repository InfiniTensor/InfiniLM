#ifndef __UTILS_H__
#define __UTILS_H__

#include "infiniop/tensor_descriptor.h"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

/* This file contains some useful macros and helper functions */

#define ROUND_UP_DIV(x, y) ((x + y - 1) / y)

#define CHECK_ERROR(call, target, errCode)                                 \
    do {                                                                   \
        if (auto value = (call); value == (target)) {                      \
            std::cerr << "Error: expected " << (target) << " but got "     \
                      << value << " in file " << __FILE__ << ", function " \
                      << __func__ << ", line " << __LINE__ << std::endl;   \
            return (errCode);                                              \
        }                                                                  \
    } while (0)

#define CREATE_CHECK_ERROR(expr, value, target, errCode) \
    expr;                                                \
    CHECK_ERROR(value, target, errCode)

#define CHECK_STATUS(call, target)                                         \
    do {                                                                   \
        if (auto value = (call); value != (target)) {                      \
            std::cerr << "Error: expected " << (target) << " but got "     \
                      << value << " in file " << __FILE__ << ", function " \
                      << __func__ << ", line " << __LINE__ << std::endl;   \
            return value;                                                  \
        }                                                                  \
    } while (0)

inline std::vector<int64_t> getByteStrides(infiniopTensorDescriptor_t desc) {
    std::vector<int64_t> strides(desc->ndim);
    for (uint64_t i = 0; i < desc->ndim; i++) {
        strides[i] = desc->strides[i] * infiniSizeof(desc->dtype);
    }
    return strides;
}

inline size_t getByteSize(infiniopTensorDescriptor_t desc) {
    size_t size = 1;
    for (size_t i = 0; i < desc->ndim; i++) {
        size *= desc->shape[i];
    }
    return size * infiniSizeof(desc->dtype);
}

// calculate the broadcasted shape for two tensors
inline bool getBroadcastShape(const uint64_t *shape1, uint64_t ndim1,
                              const uint64_t *shape2, uint64_t ndim2,
                              uint64_t *broadcast_shape,
                              uint64_t *padded_shape1, uint64_t *padded_shape2,
                              uint64_t max_rank) {
    // prepending and initializing
    std::fill(padded_shape1, padded_shape1 + max_rank, 1);
    std::fill(padded_shape2, padded_shape2 + max_rank, 1);
    std::copy(shape1, shape1 + ndim1, padded_shape1 + max_rank - ndim1);
    std::copy(shape2, shape2 + ndim2, padded_shape2 + max_rank - ndim2);

    // compute broadcasted shape
    for (size_t i = 0; i < max_rank; ++i) {
        if (padded_shape1[i] == padded_shape2[i] || padded_shape1[i] == 1 || padded_shape2[i] == 1) {
            broadcast_shape[i] = std::max(padded_shape1[i], padded_shape2[i]);
        } else {
            return false;
        }
    }

    return true;
}

// check if the shape of tensor c is valid after broadcasting tensors a and b
// and also get the broadcasted shapes
inline bool isValidBroadcastShape(infiniopTensorDescriptor_t a,
                                  infiniopTensorDescriptor_t b,
                                  infiniopTensorDescriptor_t c,
                                  uint64_t broadcast_ndim) {
    std::vector<uint64_t> broadcast_shape_(broadcast_ndim),
        padded_shape1_(broadcast_ndim), padded_shape2_(broadcast_ndim);
    auto broadcast_shape = broadcast_shape_.data(),
         padded_shape1 = padded_shape1_.data(),
         padded_shape2 = padded_shape2_.data();
    if (broadcast_ndim != c->ndim || !getBroadcastShape(a->shape, a->ndim, b->shape, b->ndim, broadcast_shape, padded_shape1, padded_shape2, broadcast_ndim)) {
        return false;
    }
    return std::equal(broadcast_shape, broadcast_shape + broadcast_ndim,
                      c->shape);
}

// check if the shape of tensor src can be validly broadcasted to that of the
// tensor dst
inline bool isValidBroadcastShape(infiniopTensorDescriptor_t dst,
                                  infiniopTensorDescriptor_t src) {
    if (dst->ndim < src->ndim) {
        return false;
    }
    std::vector<size_t> padded_shape_(dst->ndim);
    auto padded_shape = padded_shape_.data();
    std::fill(padded_shape, padded_shape + dst->ndim, 1);
    std::copy(src->shape, src->shape + src->ndim,
              padded_shape + dst->ndim - src->ndim);
    for (size_t i = 0; i < dst->ndim; ++i) {
        if (padded_shape[i] != dst->shape[i] && padded_shape[i] != 1) {
            return false;
        }
    }
    return true;
}

// check if the shape of tensor c is valid after broadcasting tensors a and b
inline bool isValidBroadcastShape(infiniopTensorDescriptor_t a,
                                  infiniopTensorDescriptor_t b,
                                  infiniopTensorDescriptor_t c) {
    return isValidBroadcastShape(a, b, c, std::max(a->ndim, b->ndim));
}

// permute the dimensions of a tensor descriptor
inline infiniopTensorDescriptor_t permute(infiniopTensorDescriptor_t desc,
                                          const std::vector<size_t> &order) {
    size_t ndim = desc->ndim;
    if (order.size() != ndim) {
        return nullptr;
    }
    size_t *shape = new size_t[ndim];
    int64_t *strides = new int64_t[ndim];
    for (size_t i = 0; i < ndim; i++) {
        if (std::find(order.begin(), order.end(), i) == order.end()) {
            return nullptr;
        }
        shape[i] = desc->shape[order[i]];
        strides[i] = desc->strides[order[i]];
    }
    return new InfiniopTensorDescriptor{desc->dtype, ndim, shape, strides};
}

// check if the dimensions [dim_start, dim_end] of a tensor descriptor are
// contiguous
inline bool isContiguous(const infiniopTensorDescriptor_t &desc,
                         size_t dim_start, size_t dim_end) {
    for (size_t i = dim_start + 1; i <= dim_end; i++) {
        if (desc->strides[i - 1] != static_cast<int64_t>(desc->shape[i]) * desc->strides[i]) {
            return false;
        }
    }
    return true;
}

inline bool isContiguous(const infiniopTensorDescriptor_t &desc) {
    if (desc->ndim == 0) {
        return true;
    }
    return isContiguous(desc, 0, desc->ndim - 1);
}

// merge the dimensions [dim_start, dim_end] of a tensor descriptor
inline infiniopTensorDescriptor_t dimMerge(infiniopTensorDescriptor_t desc,
                                           size_t dim_start, size_t dim_end) {
    size_t ndim = desc->ndim;
    if (dim_start > dim_end || dim_end >= ndim) {
        return nullptr;
    }

    size_t new_ndim = ndim - (dim_end - dim_start);
    size_t *new_shape = new size_t[new_ndim];
    int64_t *new_strides = new int64_t[new_ndim];
    size_t index = 0;
    for (size_t i = 0; i < dim_start; i++) {
        new_shape[index] = desc->shape[i];
        new_strides[index] = desc->strides[i];
        index++;
    }
    if (!isContiguous(desc, dim_start, dim_end)) {
        return nullptr;
    }
    new_shape[index] = 1;
    for (size_t i = dim_start; i <= dim_end; i++) {
        new_shape[index] *= desc->shape[i];
    }
    new_strides[index] = desc->strides[dim_end];
    index++;
    for (size_t i = dim_end + 1; i < ndim; i++) {
        new_shape[index] = desc->shape[i];
        new_strides[index] = desc->strides[i];
        index++;
    }
    return new InfiniopTensorDescriptor{desc->dtype, new_ndim, new_shape,
                                        new_strides};
}

// split the dimension dim of a tensor descriptor into multiple dimensions
inline infiniopTensorDescriptor_t dimSplit(infiniopTensorDescriptor_t desc,
                                           size_t dim,
                                           const std::vector<size_t> &dims) {
    size_t ndim = desc->ndim;
    if (desc->shape[dim] != std::accumulate(dims.begin(), dims.end(), (size_t)1, std::multiplies{})) {
        return nullptr;
    }
    size_t new_ndim = ndim + dims.size() - 1;
    size_t *new_shape = new size_t[new_ndim];
    int64_t *new_strides = new int64_t[new_ndim];
    size_t index = 0;
    for (size_t i = 0; i < dim; i++) {
        new_shape[index] = desc->shape[i];
        new_strides[index] = desc->strides[i];
        index++;
    }
    for (size_t i = 0; i < dims.size(); i++) {
        new_shape[index] = dims[i];
        new_strides[index] = desc->strides[dim] * desc->shape[dim] / std::accumulate(dims.begin(), dims.begin() + i + 1, (size_t)1, std::multiplies<size_t>());
        index++;
    }
    for (size_t i = dim + 1; i < ndim; i++) {
        new_shape[index] = desc->shape[i];
        new_strides[index] = desc->strides[i];
        index++;
    }
    return new InfiniopTensorDescriptor{desc->dtype, new_ndim, new_shape,
                                        new_strides};
}

#endif // __UTILS_H__
