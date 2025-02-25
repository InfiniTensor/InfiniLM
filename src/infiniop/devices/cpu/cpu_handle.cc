#include "cpu_handle.h"

infiniStatus_t createCpuHandle(infiniopCpuHandle_t *handle_ptr) {
    *handle_ptr = new InfiniopHandle{INFINI_DEVICE_CPU, 0};
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t destroyCpuHandle(infiniopCpuHandle_t handle) {
    delete handle;
    return INFINI_STATUS_SUCCESS;
}
