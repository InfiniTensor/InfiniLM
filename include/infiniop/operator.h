#ifndef __INFINIOP_OPERATOR___
#define __INFINIOP_OPERATOR___

#include "./handle.h"
#include "./tensor_descriptor.h"

// Base descriptor for all operators
typedef struct InfiniopDescriptor {
    infiniDevice_t device;
    int device_id;
} InfiniopDescriptor;

#endif //__INFINIOP_OPERATOR___
