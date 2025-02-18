#ifndef __INFINIOP_COMMON_CUDA_H__
#define __INFINIOP_COMMON_CUDA_H__

#define MAX_THREADS_PER_BLOCK 1024
#define MAX_WARP_PER_BLOCK 32
#define WARP_SIZE 32

#include <iostream>

#define CHECK_CUDA_OR_RETURN(call, errorCode)                                 \
    do {                                                                      \
        if (auto status = call; status != cudaSuccess) {                      \
            std::cerr << "CUDA error: " << cudaGetErrorString(status)         \
                      << " in file " << __FILE__ << ", function " << __func__ \
                      << ", line " << __LINE__ << std::endl;                  \
            return errorCode;                                                 \
        }                                                                     \
    } while (0)

#define CHECK_CUDA(call) CHECK_CUDA_OR_RETURN(call, INFINIOP_STATUS_INTERNAL_ERROR)

#define CHECK_CUDNN(call)                                                     \
    do {                                                                      \
        if (auto status = call; status != CUDNN_STATUS_SUCCESS) {             \
            std::cerr << "CUDNN error: " << cudnnGetErrorString(status)       \
                      << " in file " << __FILE__ << ", function " << __func__ \
                      << ", line " << __LINE__ << std::endl;                  \
            return INFINIOP_STATUS_INTERNAL_ERROR;                            \
        }                                                                     \
    } while (0)

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
void use_cublas(std::shared_ptr<Pool<cublasHandle_t>> cublas_handle_pool, int device_id, cudaStream_t stream, T const &f) {
    auto handle = cublas_handle_pool->pop();
    if (!handle) {
        cublasCreate(&(*handle));
    }
    cublasSetStream(*handle, (cudaStream_t)stream);
    f(*handle);
    cublas_handle_pool->push(std::move(*handle));
}

template <typename T>
cudnnStatus_t use_cudnn(std::shared_ptr<Pool<cudnnHandle_t>> cudnn_handle_pool, int device_id, cudaStream_t stream, T const &f) {
    auto handle = cudnn_handle_pool->pop();
    if (!handle) {
        cudnnCreate(&(*handle));
    }
    cudnnSetStream(*handle, stream);
    cudnnStatus_t status = f(*handle);
    cudnn_handle_pool->push(std::move(*handle));
    return status;
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
    size_t flat_index, size_t ndim, ptrdiff_t const *broadcasted_strides,
    ptrdiff_t const *target_strides) {
    size_t res = 0;
    for (size_t i = 0; i < ndim; ++i) {
        res += flat_index / broadcasted_strides[i] * target_strides[i];
        flat_index %= broadcasted_strides[i];
    }
    return res;
}

// get the memory offset of the given element in a tensor given its flat index
inline __device__ __host__ size_t indexToOffset(size_t flat_index, size_t ndim,
                                                size_t const *shape,
                                                ptrdiff_t const *strides) {
    size_t res = 0;
    for (size_t i = ndim; i-- > 0;) {
        res += (flat_index % shape[i]) * strides[i];
        flat_index /= shape[i];
    }
    return res;
}

#endif // __INFINIOP_COMMON_CUDA_H__
