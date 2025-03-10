#include "kunlun_handle.h"

namespace device::kunlun {

Handle::Handle(infiniDevice_t device, int device_id)
    : InfiniopHandle{device, device_id},
      _internal(std::make_shared<Handle::Internal>()) {}

auto Handle::internal() const -> const std::shared_ptr<Internal> & {
    return _internal;
}

template <typename T>
using Fn = std::function<void(T)>;

void Handle::Internal::use_xdnn(kunlunStream_t stream, const Fn<xdnnHandle_t> &f) const {
    auto handle = dnn_handles.pop();
    if (!handle) {
        *handle = xdnn::create_context();
    }
    (*handle)->set_stream(stream);
    f(*handle);
    dnn_handles.push(std::move(*handle));
}

infiniStatus_t Handle::create(InfiniopHandle **handle_ptr, int device_id) {
    *handle_ptr = new Handle(INFINI_DEVICE_KUNLUN, device_id);
    return INFINI_STATUS_SUCCESS;
}

} // namespace device::kunlun
