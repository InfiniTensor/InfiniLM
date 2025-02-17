#include "../pool.h"
#include "common_bang.h"
#include <memory>

infiniopStatus_t createBangHandle(infiniopBangHandle_t *handle_ptr) {
    int device_id = 0;
    if (cnrtGetDevice(&device_id) != cnrtSuccess) {
        return INFINIOP_STATUS_BAD_DEVICE;
    }

    auto pool = std::make_shared<Pool<cnnlHandle_t>>();
    cnnlHandle_t handle;
    cnnlCreate(&handle);
    pool->push(std::move(handle));

    *handle_ptr = new InfiniopBangHandle{INFINI_DEVICE_CAMBRICON, device_id,
                                         std::move(pool)};

    return INFINIOP_STATUS_SUCCESS;
}

infiniopStatus_t destroyBangHandle(infiniopBangHandle_t handle) {
    delete handle;
    return INFINIOP_STATUS_SUCCESS;
}
