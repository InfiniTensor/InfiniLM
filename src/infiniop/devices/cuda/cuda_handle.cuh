#ifndef __INFINIOP_CUDA_HANDLE_CUH__
#define __INFINIOP_CUDA_HANDLE_CUH__

#include "../../../utils.h"
#include "../pool.h"
#include "cuda_functions.cuh"
#include "cuda_handle.h"
#include <cublas_v2.h>
#include <functional>

#define CHECK_CUBLAS(API) CHECK_INTERNAL(API, CUBLAS_STATUS_SUCCESS)
#define CHECK_CUDNN(API) CHECK_INTERNAL(API, CUDNN_STATUS_SUCCESS)

namespace device::cuda {

class Handle::Internal {
    Pool<cublasHandle_t> blas_handles;
    Pool<cudnnHandle_t> dnn_handles;

    int _warp_size,
        _max_threads_per_block,
        _block_size[3],
        _grid_size[3];

    template <typename T>
    using Fn = std::function<infiniStatus_t(T)>;

public:
    Internal(int);

    infiniStatus_t useCublas(cudaStream_t stream, const Fn<cublasHandle_t> &f) const;
    infiniStatus_t useCudnn(cudaStream_t stream, const Fn<cudnnHandle_t> &f) const;

    int warpSize() const;
    int maxThreadsPerBlock() const;
    int blockSizeX() const;
    int blockSizeY() const;
    int blockSizeZ() const;
    int gridSizeX() const;
    int gridSizeY() const;
    int gridSizeZ() const;
};

} // namespace device::cuda

#endif // __INFINIOP_CUDA_HANDLE_CUH__
