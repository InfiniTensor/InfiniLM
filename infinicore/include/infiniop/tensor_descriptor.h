#ifndef __INFINIOP_TENSOR_DESCRIPTOR_API_H__
#define __INFINIOP_TENSOR_DESCRIPTOR_API_H__

#include "../infinicore.h"

struct InfiniopTensorDescriptor;

typedef struct InfiniopTensorDescriptor *infiniopTensorDescriptor_t;

__C __export infiniStatus_t infiniopCreateTensorDescriptor(infiniopTensorDescriptor_t *desc_ptr, size_t ndim, const size_t *shape, const ptrdiff_t *strides, infiniDtype_t dtype);

__C __export infiniStatus_t infiniopDestroyTensorDescriptor(infiniopTensorDescriptor_t desc);

#endif // __INFINIOP_TENSOR_DESCRIPTOR__
