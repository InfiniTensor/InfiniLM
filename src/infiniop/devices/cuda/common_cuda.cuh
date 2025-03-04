#ifndef __INFINIOP_COMMON_CUDA_H__
#define __INFINIOP_COMMON_CUDA_H__

#define MAX_THREADS_PER_BLOCK 1024
#define MAX_WARP_PER_BLOCK 32
#define WARP_SIZE 32

#include "../../utils.h"
#include <iostream>

#define CHECK_CUDA_OR_RETURN(API, ERROR) CHECK_API_OR(API, cudaSuccess, return ERROR)

#define CHECK_CUDA(API) CHECK_INTERNAL(API, cudaSuccess)

#define CHECK_CUDNN(API) CHECK_INTERNAL(API, CUDNN_STATUS_SUCCESS)

#include "../pool.h"
#include "cuda_handle.h"
#include "infinicore.h"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <memory>

struct InfiniopCudaHandle {
    infiniDevice_t device;
    int device_id;
    std::shared_ptr<Pool<cublasHandle_t>> cublas_handle_pool;
    std::shared_ptr<Pool<cudnnHandle_t>> cudnn_handle_pool;
    cudaDeviceProp prop;
    int compute_capability_major;
    int compute_capability_minor;
};

template <typename T>
void use_cublas(std::shared_ptr<Pool<cublasHandle_t>> &pool, cudaStream_t stream, const T &f) {
    auto handle = pool->pop();
    if (!handle) {
        cublasCreate(&(*handle));
    }
    cublasSetStream(*handle, stream);
    f(*handle);
    pool->push(std::move(*handle));
}

template <typename T>
void use_cudnn(std::shared_ptr<Pool<cudnnHandle_t>> &pool, cudaStream_t stream, const T &f) {
    auto handle = pool->pop();
    if (!handle) {
        cudnnCreate(&(*handle));
    }
    cudnnSetStream(*handle, stream);
    f(*handle);
    pool->push(std::move(*handle));
}

inline cudnnDataType_t getCudnnDtype(infiniDtype_t dt) {
    switch (dt) {
    case INFINI_DTYPE_F16:
        return CUDNN_DATA_HALF;
    case INFINI_DTYPE_F32:
        return CUDNN_DATA_FLOAT;
    case INFINI_DTYPE_F64:
        return CUDNN_DATA_DOUBLE;
    case INFINI_DTYPE_BF16:
        return CUDNN_DATA_BFLOAT16;
    case INFINI_DTYPE_I8:
        return CUDNN_DATA_INT8;
    case INFINI_DTYPE_I32:
        return CUDNN_DATA_INT32;
    case INFINI_DTYPE_I64:
        return CUDNN_DATA_INT64;
    case INFINI_DTYPE_U8:
        return CUDNN_DATA_UINT8;
    default:
        return CUDNN_DATA_FLOAT;
    }
}

// return the memory offset of original tensor, given the flattened index of
// broadcasted tensor
inline __device__ __host__ size_t indexToReducedOffset(
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
inline __device__ __host__ size_t indexToOffset(
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

#endif // __INFINIOP_COMMON_CUDA_H__
