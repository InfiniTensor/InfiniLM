#ifndef __INFINIOP_CUDA_HANDLE_H__
#define __INFINIOP_CUDA_HANDLE_H__

#include "../../handle.h"
#include <memory>

namespace device::nvidia {

struct Handle : public InfiniopHandle {
    Handle(int device_id);
    class Internal;
    auto internal() const -> const std::shared_ptr<Internal> &;

public:
    static infiniStatus_t create(InfiniopHandle **handle_ptr, int device_id);

private:
    std::shared_ptr<Internal> _internal;
};

} // namespace device::nvidia

#endif // __INFINIOP_CUDA_HANDLE_H__
