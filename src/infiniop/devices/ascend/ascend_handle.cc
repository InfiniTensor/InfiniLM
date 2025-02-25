#include "common_ascend.h"

infiniStatus_t createAscendHandle(infiniopAscendHandle_t *handle_ptr) {
    int device_id = 0;
    auto ret = aclrtGetDevice(&device_id);
    CHECK_RET(ret == ACL_SUCCESS,
              return INFINI_STATUS_DEVICE_NOT_INITIALIZED);

    *handle_ptr = new InfiniopAscendHandle{INFINI_DEVICE_ASCEND, device_id};

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t destroyAscendHandle(infiniopAscendHandle_t handle_ptr) {
    delete handle_ptr;
    return INFINI_STATUS_SUCCESS;
}
