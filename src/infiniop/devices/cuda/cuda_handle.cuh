#ifndef __INFINIOP_CUDA_INTERNAL_H__
#define __INFINIOP_CUDA_INTERNAL_H__

#include "../pool.h"
#include "cuda_handle.h"
#include <cublas_v2.h>
#include <cudnn.h>
#include <functional>

namespace device::cuda {

class Handle::Internal {
    Pool<cublasHandle_t> blas_handles;
    Pool<cudnnHandle_t> dnn_handles;

public:
    void use_cublas(cudaStream_t stream, const std::function<void(cublasHandle_t)> &f) const;
    void use_cudnn(cudaStream_t stream, const std::function<void(cudnnHandle_t)> &f) const;
};

cudnnDataType_t getCudnnDtype(infiniDtype_t dt);

// return the memory offset of original tensor, given the flattened index of broadcasted tensor
__forceinline__ __device__ __host__ size_t
indexToReducedOffset(
    size_t flat_index,
    size_t ndim,
    const ptrdiff_t *broadcasted_strides,
    const ptrdiff_t *target_strides) {
    size_t res = 0;
    for (size_t i = 0; i < ndim; ++i) {
        res += flat_index / broadcasted_strides[i] * target_strides[i];
        flat_index %= broadcasted_strides[i];
    }
    return res;
}

// get the memory offset of the given element in a tensor given its flat index
__forceinline__ __device__ __host__ size_t
indexToOffset(
    size_t flat_index,
    size_t ndim,
    const size_t *shape,
    const ptrdiff_t *strides) {
    size_t res = 0;
    for (size_t i = ndim; i-- > 0;) {
        res += (flat_index % shape[i]) * strides[i];
        flat_index /= shape[i];
    }
    return res;
}

} // namespace device::cuda

#endif // __INFINIOP_CUDA_INTERNAL_H__
