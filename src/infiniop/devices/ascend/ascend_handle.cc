#include "common_ascend.h"

infiniopStatus_t createAscendHandle(infiniopAscendHandle_t *handle_ptr, int device_id) {
    uint32_t device_count;
    aclrtGetDeviceCount(&device_count);
    if (device_id >= static_cast<int>(device_count)) {
        return INFINIOP_STATUS_BAD_DEVICE;
    }

    auto ret = aclrtSetDevice(device_id);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret));

    *handle_ptr = new InfiniopAscendHandle{INFINI_DEVICE_ASCEND, device_id};

    return INFINIOP_STATUS_SUCCESS;
}

infiniopStatus_t deleteAscendHandle(infiniopAscendHandle_t handle_ptr) {
    delete handle_ptr;

    return INFINIOP_STATUS_SUCCESS;
}
