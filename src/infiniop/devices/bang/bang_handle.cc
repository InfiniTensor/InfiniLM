#include "bang_handle.h"

infiniopStatus_t createBangHandle(infiniopBangHandle_t *handle_ptr, int device_id) {
    unsigned int device_count;
    cnrtGetDeviceCount(&device_count);
    if (device_id >= static_cast<int>(device_count)) {
        return INFINIOP_STATUS_BAD_DEVICE;
    }

    auto pool = std::make_shared<Pool<cnnlHandle_t>>();
    if (cnrtSetDevice(device_id) != cnrtSuccess){
        return INFINIOP_STATUS_BAD_DEVICE;
    }
    cnnlHandle_t handle;
    cnnlCreate(&handle);
    pool->push(std::move(handle));

    *handle_ptr = new InfiniopBangHandle{INFINI_DEVICE_CAMBRICON, device_id, std::move(pool)};

    return INFINIOP_STATUS_SUCCESS;
}
