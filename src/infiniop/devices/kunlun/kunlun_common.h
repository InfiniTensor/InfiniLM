#include "../pool.h"
#include "kunlun_handle.h"
#include <xpu/runtime.h>
#include <xpu/runtime_ex.h>
#include <xpu/xdnn.h>

namespace xdnn = baidu::xpu::api;

typedef XPUStream kunlunStream_t;
typedef XPUEvent kunlunEvent_t;
typedef xdnn::Context *xdnnHandle_t;

#define CHECK_KUNLUN(API) CHECK_INTERNAL(API, XPU_SUCCESS)

namespace device::kunlun {

class Handle::Internal {
    Pool<xdnnHandle_t> dnn_handles;
    template <typename T>
    using Fn = std::function<infiniStatus_t(T)>;

public:
    infiniStatus_t useXdnn(kunlunStream_t stream, const Fn<xdnnHandle_t> &f) const;
};

} // namespace device::kunlun
