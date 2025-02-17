#include "common_ascend.h"

infiniopStatus_t createAscendHandle(infiniopAscendHandle_t *handle_ptr) {
    int device_id = 0;
    auto ret = aclrtGetDevice(&device_id);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_ERROR("aclrtGetDevice failed. ERROR: %d\n", ret));

    *handle_ptr = new InfiniopAscendHandle{INFINI_DEVICE_ASCEND, device_id};

    return INFINIOP_STATUS_SUCCESS;
}

infiniopStatus_t destroyAscendHandle(infiniopAscendHandle_t handle_ptr) {
    delete handle_ptr;
    return INFINIOP_STATUS_SUCCESS;
}
