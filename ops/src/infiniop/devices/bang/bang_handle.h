#ifndef __INFINIOP_BANG_HANDLE_H__
#define __INFINIOP_BANG_HANDLE_H__

#include "../../handle.h"
#include <memory>

namespace device::bang {

struct Handle : public InfiniopHandle {
    class Internal;
    auto internal() const -> const std::shared_ptr<Internal> &;

protected:
    Handle(infiniDevice_t device, int device_id);

private:
    std::shared_ptr<Internal> _internal;
};

namespace cambricon {

class Handle : public bang::Handle {
    Handle(int device_id);

public:
    static infiniStatus_t create(InfiniopHandle **handle_ptr, int device_id);
};

} // namespace cambricon

} // namespace device::bang

#endif // __INFINIOP_BANG_HANDLE_H__
