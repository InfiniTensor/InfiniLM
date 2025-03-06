#include "cuda_handle.cuh"

namespace device::cuda {

Handle::Handle(infiniDevice_t device, int device_id)
    : InfiniopHandle{device, device_id},
      _internal(std::make_shared<Handle::Internal>()) {}

auto Handle::internal() const -> const std::shared_ptr<Internal> & {
    return _internal;
}

template <typename T>
using Fn = std::function<void(T)>;

void Handle::Internal::use_cublas(cudaStream_t stream, const Fn<cublasHandle_t> &f) const {
    auto handle = blas_handles.pop();
    if (!handle) {
        cublasCreate(&(*handle));
    }
    cublasSetStream(*handle, stream);
    f(*handle);
    blas_handles.push(std::move(*handle));
}

void Handle::Internal::use_cudnn(cudaStream_t stream, const Fn<cudnnHandle_t> &f) const {
    auto handle = dnn_handles.pop();
    if (!handle) {
        cudnnCreate(&(*handle));
    }
    cudnnSetStream(*handle, stream);
    f(*handle);
    dnn_handles.push(std::move(*handle));
}

cudnnDataType_t getCudnnDtype(infiniDtype_t dt) {
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

namespace nvidia {

Handle::Handle(int device_id)
    : cuda::Handle(INFINI_DEVICE_NVIDIA, device_id) {}

infiniStatus_t Handle::create(InfiniopHandle **handle_ptr, int device_id) {
    *handle_ptr = new Handle(device_id);
    return INFINI_STATUS_SUCCESS;
}

} // namespace nvidia

} // namespace device::cuda
