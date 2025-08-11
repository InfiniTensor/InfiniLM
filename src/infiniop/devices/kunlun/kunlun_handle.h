#ifndef __INFINIOP_KUNLUN_HANDLE_H__
#define __INFINIOP_KUNLUN_HANDLE_H__

#include "../../handle.h"
#include <memory>

namespace device::kunlun {

struct Handle : public InfiniopHandle {
    class Internal;
    auto internal() const -> const std::shared_ptr<Internal> &;

    Handle(int device_id);

private:
    std::shared_ptr<Internal> _internal;

public:
    static infiniStatus_t create(InfiniopHandle **handle_ptr, int device_id);
};

} // namespace device::kunlun

#endif // __INFINIOP_KUNLUN_HANDLE_H__
