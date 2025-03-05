#include "operator.h"

infiniStatus_t infiniopGetDescriptorDeviceType(
    const InfiniopDescriptor *desc_ptr,
    infiniDevice_t *device_type) {
    *device_type = desc_ptr->device_type;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t infiniopGetDescriptorDeviceId(
    const InfiniopDescriptor *desc_ptr,
    int *device_id) {
    *device_id = desc_ptr->device_id;
    return INFINI_STATUS_SUCCESS;
}
