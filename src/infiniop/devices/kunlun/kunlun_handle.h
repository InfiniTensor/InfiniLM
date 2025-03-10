#ifndef __INFINIOP_KUNLUN_HANDLE_H__
#define __INFINIOP_KUNLUN_HANDLE_H__

#include "../../handle.h"
#include "../pool.h"
#include <functional>
#include <memory>
#include <xpu/runtime.h>
#include <xpu/runtime_ex.h>
#include <xpu/xdnn.h>

namespace xdnn = baidu::xpu::api;

typedef XPUStream kunlunStream_t;
typedef XPUEvent kunlunEvent_t;
typedef xdnn::Context *xdnnHandle_t;

namespace device::kunlun {

struct Handle : public InfiniopHandle {
    class Internal;
    auto internal() const -> const std::shared_ptr<Internal> &;

    Handle(infiniDevice_t device, int device_id);

private:
    std::shared_ptr<Internal> _internal;

public:
    static infiniStatus_t create(InfiniopHandle **handle_ptr, int device_id);
};

class Handle::Internal {
    Pool<xdnnHandle_t> dnn_handles;

public:
    void use_xdnn(kunlunStream_t stream, const std::function<void(xdnnHandle_t)> &f) const;
};

} // namespace device::kunlun

#endif // __INFINIOP_KUNLUN_HANDLE_H__
