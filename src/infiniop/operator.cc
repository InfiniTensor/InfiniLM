#include "infiniop/operator.h"

infiniopStatus_t infiniopGetDescriptorDeviceType(
    InfiniopDescriptor const *desc_ptr,
    infiniDevice_t *device_type) {
    *device_type = desc_ptr->device_type;
    return INFINIOP_STATUS_SUCCESS;
}

infiniopStatus_t infiniopGetDescriptorDeviceId(
    InfiniopDescriptor const *desc_ptr,
    int *device_id) {
    *device_id = desc_ptr->device_id;
    return INFINIOP_STATUS_SUCCESS;
}
