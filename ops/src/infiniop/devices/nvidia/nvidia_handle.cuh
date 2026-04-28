#ifndef __INFINIOP_CUDA_HANDLE_CUH__
#define __INFINIOP_CUDA_HANDLE_CUH__

#include "../../../utils.h"
#include "../pool.h"
#include "nvidia_handle.h"
#include <cublas_v2.h>
#include <functional>

#ifdef ENABLE_CUDNN_API
#include <cudnn.h>
#endif

#define CHECK_CUBLAS(API) CHECK_INTERNAL(API, CUBLAS_STATUS_SUCCESS)
#define CHECK_CUDNN(API) CHECK_INTERNAL(API, CUDNN_STATUS_SUCCESS)

namespace device::nvidia {

class Handle::Internal {
    Pool<cublasHandle_t> blas_handles;
#ifdef ENABLE_CUDNN_API
    Pool<cudnnHandle_t> dnn_handles;
#endif

    // CUDA Graph capture mode: when active, useCublas/useCudnn skip the
    // pool-pop/push and cublasSetStream calls (which break stream capture
    // semantics on iluvatar's ixblas). Instead they use a single pre-warmed
    // handle whose stream binding was set up before BeginCapture.
    mutable cublasHandle_t capture_blas_handle_ = nullptr;
#ifdef ENABLE_CUDNN_API
    mutable cudnnHandle_t capture_dnn_handle_ = nullptr;
#endif

    int _warp_size,
        _max_threads_per_block,
        _block_size[3],
        _grid_size[3];

    template <typename T>
    using Fn = std::function<infiniStatus_t(T)>;

public:
    Internal(int);
    ~Internal();

    infiniStatus_t useCublas(cudaStream_t stream, const Fn<cublasHandle_t> &f) const;
#ifdef ENABLE_CUDNN_API
    infiniStatus_t useCudnn(cudaStream_t stream, const Fn<cudnnHandle_t> &f) const;
#endif

    // Process-global capture-mode flag. Assumes graph capture is single-flight
    // and non-reentrant: Graph::instantiate() pairs setCaptureMode(true)/(false)
    // around a single cudaStreamBeginCapture/EndCapture window.
    static bool isCaptureMode();
    static void setCaptureMode(bool enabled);

    int warpSize() const;
    int maxThreadsPerBlock() const;
    int blockSizeX() const;
    int blockSizeY() const;
    int blockSizeZ() const;
    int gridSizeX() const;
    int gridSizeY() const;
    int gridSizeZ() const;
};

} // namespace device::nvidia

#endif // __INFINIOP_CUDA_HANDLE_CUH__
