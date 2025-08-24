#include "ascend_handle.h"

namespace device::ascend {

Handle::Handle(int device_id)
    : InfiniopHandle{INFINI_DEVICE_ASCEND, device_id} {}

infiniStatus_t Handle::create(InfiniopHandle **Handle_ptr, int device_id) {
    *Handle_ptr = new Handle(device_id);

    return INFINI_STATUS_SUCCESS;
}

} // namespace device::ascend
