#include "infiniop/operator.h"

infiniopStatus_t infiniopGetDescriptorDeviceType(
    const InfiniopDescriptor *desc_ptr,
    infiniDevice_t *device_type) {
    *device_type = desc_ptr->device_type;
    return INFINIOP_STATUS_SUCCESS;
}

infiniopStatus_t infiniopGetDescriptorDeviceId(
    const InfiniopDescriptor *desc_ptr,
    int *device_id) {
    *device_id = desc_ptr->device_id;
    return INFINIOP_STATUS_SUCCESS;
}
