#include "cpu_handle.h"

namespace device::cpu {

Handle::Handle() : InfiniopHandle{INFINI_DEVICE_CPU, 0} {}

infiniStatus_t Handle::create(InfiniopHandle **handle_ptr, int) {
    *handle_ptr = new Handle{};
    return INFINI_STATUS_SUCCESS;
}

} // namespace device::cpu
