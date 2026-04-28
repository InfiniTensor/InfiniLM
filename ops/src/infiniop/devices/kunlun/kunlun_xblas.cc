#include "kunlun_xblas.h"

namespace device::kunlun::blas {

Handle::Handle(int device_id)
    : InfiniopHandle{INFINI_DEVICE_KUNLUN, device_id},
      _internal(std::make_shared<Handle::Internal>()) {}

auto Handle::internal() const -> const std::shared_ptr<Internal> & {
    return _internal;
}

infiniStatus_t Handle::create(InfiniopHandle **handle_ptr, int device_id) {
    *handle_ptr = new Handle(device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Handle::Internal::useCublas(cudaStream_t stream, const Fn<cublasHandle_t> &f) const {

    auto handle = blas_handles.pop();
    if (!handle) {
        CHECK_CUBLAS(cublasCreate(&(*handle)));
    }
    CHECK_CUBLAS(cublasSetStream(*handle, stream));
    CHECK_STATUS(f(*handle));
    blas_handles.push(std::move(*handle));
    return INFINI_STATUS_SUCCESS;
}

} // namespace device::kunlun::blas
