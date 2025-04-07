#include "common_musa.h"

namespace device::musa {
Handle::Handle(infiniDevice_t device, int device_id)
    : InfiniopHandle{device, device_id},
      _internal(std::make_shared<Handle::Internal>()) {}

Handle::Handle(int device_id) : Handle(INFINI_DEVICE_MOORE, device_id) {}

auto Handle::internal() const -> const std::shared_ptr<Internal> & {
    return _internal;
}

infiniStatus_t Handle::Internal::useMublas(musaStream_t stream, const Fn<mublasHandle_t> &f) const {
    std::unique_ptr<mublasHandle_t> handle;
    auto opt_handle = mublas_handles.pop();
    if (opt_handle.has_value()) {
        handle = std::move(*opt_handle);
    } else {
        handle = std::make_unique<mublasHandle_t>();
        CHECK_MUBLAS(mublasCreate(&(*handle)));
    }
    CHECK_MUBLAS(mublasSetStream(*handle, stream));
    CHECK_STATUS(f(*handle));
    mublas_handles.push(std::move(handle));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Handle::Internal::useMudnn(musaStream_t stream, const Fn<::musa::dnn::Handle &> &f) const {
    std::unique_ptr<::musa::dnn::Handle> handle;
    auto opt_handle = mudnn_handles.pop();
    if (opt_handle.has_value()) {
        handle = std::move(*opt_handle);
    } else {
        handle = std::make_unique<::musa::dnn::Handle>();
    }
    CHECK_MUDNN(handle->SetStream(stream));
    CHECK_STATUS(f(*handle));
    mudnn_handles.push(std::move(handle));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Handle::create(InfiniopHandle **handle_ptr, int device_id) {
    *handle_ptr = new Handle(INFINI_DEVICE_MOORE, device_id);
    return INFINI_STATUS_SUCCESS;
}

} // namespace device::musa
