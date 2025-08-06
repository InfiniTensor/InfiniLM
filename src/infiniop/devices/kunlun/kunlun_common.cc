#include "kunlun_common.h"
#include "../../../utils.h"
#include <functional>

namespace device::kunlun {

infiniStatus_t Handle::Internal::useXdnn(kunlunStream_t stream, const Fn<xdnnHandle_t> &f) const {
    auto handle = dnn_handles.pop();
    if (!handle) {
        *handle = xdnn::create_context();
    }
    (*handle)->set_stream(stream);
    CHECK_STATUS(f(*handle));
    dnn_handles.push(std::move(*handle));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Handle::create(InfiniopHandle **handle_ptr, int device_id) {
    *handle_ptr = new Handle(device_id);
    return INFINI_STATUS_SUCCESS;
}
} // namespace device::kunlun
