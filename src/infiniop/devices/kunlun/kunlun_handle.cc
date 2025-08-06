#include "kunlun_common.h"

namespace device::kunlun {

Handle::Handle(int device_id)
    : InfiniopHandle{INFINI_DEVICE_KUNLUN, device_id},
      _internal(std::make_shared<Handle::Internal>()) {}

auto Handle::internal() const -> const std::shared_ptr<Internal> & {
    return _internal;
}

} // namespace device::kunlun
