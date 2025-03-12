#include "common_musa.h"

namespace device::musa {
Handle::Handle(infiniDevice_t device, int device_id)
    : InfiniopHandle{device, device_id},
      _internal(std::make_shared<Handle::Internal>()) {}

auto Handle::internal() const -> const std::shared_ptr<Internal> & {
    return _internal;
}

infiniStatus_t Handle::Internal::useMublas(MUstream stream, const Fn<mublasHandle_t> &f) const {
    mublasHandle_t *handle = mublas_handles.pop();
    if (!handle) {
        handle = new mublasHandle_t;
        CHECK_MUBLAS(mublasCreate(handle));
    }
    CHECK_MUBLAS(mublasSetStream(*handle, stream));
    CHECK_STATUS(f(*handle));
    mublas_handles.push(handle);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Handle::Internal::useMudnn(musaStream_t stream, const Fn<::musa::dnn::Handle &> &f) const {
    ::musa::dnn::Handle *handle = mudnn_handles.pop();
    if (!handle) {
        handle = new ::musa::dnn::Handle();
    }
    CHECK_MUDNN(handle->SetStream(stream));
    CHECK_STATUS(f(*handle));
    mudnn_handles.push(handle);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Handle::create(InfiniopHandle **handle_ptr, int device_id) {
    *handle_ptr = new Handle(INFINI_DEVICE_MOORE, device_id);
    return INFINI_STATUS_SUCCESS;
}

} // namespace device::musa
