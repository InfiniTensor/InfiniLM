#include "common_ascend.h"

infiniStatus_t createAscendHandle(infiniopAscendHandle_t *handle_ptr) {
    int device_id = 0;
    CHECK_ACL(aclrtGetDevice(&device_id));

    *handle_ptr = new InfiniopAscendHandle{INFINI_DEVICE_ASCEND, device_id};

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t destroyAscendHandle(infiniopAscendHandle_t handle_ptr) {
    delete handle_ptr;
    return INFINI_STATUS_SUCCESS;
}
