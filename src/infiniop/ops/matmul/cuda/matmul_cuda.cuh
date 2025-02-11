#ifndef __INFINIOP_MATMUL_CUDA_H__
#define __INFINIOP_MATMUL_CUDA_H__

#include "matmul_cuda_api.h"
#include "../../../devices/cuda/common_cuda.cuh"
#include <memory>
#include "../blas.h"

typedef struct InfiniopMatmulCudaDescriptor {
    infiniDevice_t device;
    infiniDtype_t dtype;
    int device_id;
    MatmulInfo info;
    std::shared_ptr<Pool<cublasHandle_t>> cublas_handles_t;
} InfiniopMatmulCudaDescriptor;

#endif// __INFINIOP_MATMUL_CUDA_H__
