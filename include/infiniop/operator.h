#ifndef __INFINIOP_OPERATOR___
#define __INFINIOP_OPERATOR___

#include "handle.h"
#include "tensor_descriptor.h"

// Base descriptor for all operators
typedef struct InfiniopDescriptor {
    infiniDevice_t device_type;
    int device_id;
} InfiniopDescriptor;

__C __export infiniStatus_t infiniopGetDescriptorDeviceType(const InfiniopDescriptor *desc_ptr, infiniDevice_t *device_type);
__C __export infiniStatus_t infiniopGetDescriptorDeviceId(const InfiniopDescriptor *desc_ptr, int *device_id);

#endif //__INFINIOP_OPERATOR___
