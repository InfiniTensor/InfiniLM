#ifndef __INFINIOP_CUDA_HANDLE_H__
#define __INFINIOP_CUDA_HANDLE_H__

#include "../../handle.h"
#include <memory>

namespace device {

namespace nvidia {

struct Handle : public InfiniopHandle {
    class Internal;
    auto internal() const -> const std::shared_ptr<Internal> &;

protected:
    Handle(infiniDevice_t device, int device_id);

public:
    static infiniStatus_t create(InfiniopHandle **handle_ptr, int device_id);

private:
    std::shared_ptr<Internal> _internal;
};

} // namespace nvidia

namespace iluvatar {

struct Handle : public nvidia::Handle {
    Handle(int device_id);

public:
    static infiniStatus_t create(InfiniopHandle **handle_ptr, int device_id);
};

} // namespace iluvatar

namespace ali {

struct Handle : public nvidia::Handle {
    Handle(int device_id);

public:
    static infiniStatus_t create(InfiniopHandle **handle_ptr, int device_id);
};

} // namespace ali

namespace qy {

struct Handle : public nvidia::Handle {
    Handle(int device_id);

public:
    static infiniStatus_t create(InfiniopHandle **handle_ptr, int device_id);
};

} // namespace qy

namespace hygon {

struct Handle : public nvidia::Handle {
    Handle(int device_id);

public:
    static infiniStatus_t create(InfiniopHandle **handle_ptr, int device_id);
};

} // namespace hygon

} // namespace device

#endif // __INFINIOP_CUDA_HANDLE_H__
