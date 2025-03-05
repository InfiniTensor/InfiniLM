#ifndef __INFINIOP_COMMON_KUNLUN_H__
#define __INFINIOP_COMMON_KUNLUN_H__

#include "../../../utils.h"
#include "../pool.h"
#include "infinicore.h"
#include "kunlun_handle.h"
#include "xpu/runtime.h"
#include "xpu/runtime_ex.h"
#include "xpu/xdnn.h"
#include <memory>

namespace xdnn = baidu::xpu::api;
typedef xdnn::Context *xdnnHandle_t;
typedef XPUStream KunlunStream_t;

#define CHECK_KUNLUN(call) CHECK_INTERNAL(call, XPU_SUCCESS)

struct InfiniopKunlunHandle {
    infiniDevice_t device;
    int device_id;
    std::shared_ptr<Pool<xdnnHandle_t>> xdnn_handle_pool;
};

template <typename T>
void use_xdnn(std::shared_ptr<Pool<xdnnHandle_t>> &pool, KunlunStream_t stream, const T &f) {
    auto handle = pool->pop();
    if (!handle) {
        *handle = xdnn::create_context();
    }
    (*handle)->set_stream(stream);
    f(*handle);
    pool->push(std::move(*handle));
}

#endif //__INFINIOP_COMMON_KUNLUN_H__
