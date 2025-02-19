#ifndef __INFINIOP_COMMON_KUNLUN_H__
#define __INFINIOP_COMMON_KUNLUN_H__

#include "../pool.h"
#include "infinicore.h"
#include "kunlun_handle.h"
#include "xpu/runtime.h"
#include "xpu/runtime_ex.h"
#include "xpu/xdnn.h"
#include <memory>

namespace xdnn = baidu::xpu::api;
typedef xdnn::Context *xdnnHandle_t;

#define CHECK_KUNLUN(call)                                             \
    {                                                                  \
        auto err = call;                                               \
        if (XPU_SUCCESS != err) {                                      \
            fprintf(stderr, "KUNLUN error in %s:%i : %s.\n", __FILE__, \
                    __LINE__, xpu_strerror(err));                      \
            return INFINIOP_STATUS_INTERNAL_ERROR;                     \
        }                                                              \
    }

struct InfiniopKunlunHandle {
    infiniDevice_t device;
    int device_id;
    std::shared_ptr<Pool<xdnnHandle_t>> xdnn_handle_pool;
};

template <typename T>
infiniopStatus_t use_xdnn(std::shared_ptr<Pool<xdnnHandle_t>> xdnn_handle_pool,
                          XPUStream stream,
                          T const &f) {
    auto handle = xdnn_handle_pool->pop();
    if (!handle) {
        *handle = xdnn::create_context();
    }
    (*handle)->set_stream(stream);
    auto ret = f(*handle);
    xdnn_handle_pool->push(std::move(*handle));
    return ret;
}

#endif
