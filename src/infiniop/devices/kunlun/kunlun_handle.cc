#include "common_kunlun.h"

infiniStatus_t createKunlunHandle(infiniopKunlunHandle_t *handle_ptr) {
    int device_id;
    CHECK_KUNLUN(xpu_current_device(&device_id));
    auto pool = std::make_shared<Pool<xdnnHandle_t>>();
    xdnnHandle_t handle = xdnn::create_context();
    pool->push(std::move(handle));

    *handle_ptr = new InfiniopKunlunHandle{
        INFINI_DEVICE_KUNLUN,
        device_id,
        std::move(pool),
    };

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t destroyKunlunHandle(infiniopKunlunHandle_t handle_ptr) {
    handle_ptr->xdnn_handle_pool = nullptr;
    delete handle_ptr;

    return INFINI_STATUS_SUCCESS;
}
