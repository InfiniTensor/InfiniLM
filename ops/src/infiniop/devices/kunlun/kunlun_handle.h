#ifndef __INFINIOP_KUNLUN_HANDLE_H__
#define __INFINIOP_KUNLUN_HANDLE_H__

#include "../../handle.h"
#include "../pool.h"
#include "kunlun_common.h"
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

class Handle::Internal {
    Pool<xdnnHandle_t> dnn_handles;
    template <typename T>
    using Fn = std::function<infiniStatus_t(T)>;

public:
    infiniStatus_t useXdnn(kunlunStream_t stream, const Fn<xdnnHandle_t> &f) const;
};

} // namespace device::kunlun

#endif // __INFINIOP_KUNLUN_HANDLE_H__
