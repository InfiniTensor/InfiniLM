#ifndef __INFINIOP_TENSOR_DESCRIPTOR__
#define __INFINIOP_TENSOR_DESCRIPTOR__

#include "../infinicore.h"

struct InfiniopTensorDescriptor {
    // Datatype
    infiniDtype_t dtype;
    // Number of dimensions
    size_t ndim;
    // Shape of the tensor, ndim elements
    size_t *shape;
    // Stride of each dimension in elements, ndim elements
    ptrdiff_t *strides;
};

typedef struct InfiniopTensorDescriptor *infiniopTensorDescriptor_t;

__C __export infiniStatus_t infiniopCreateTensorDescriptor(infiniopTensorDescriptor_t *desc_ptr, size_t ndim, const size_t *shape, const ptrdiff_t *strides, infiniDtype_t dtype);

__C __export infiniStatus_t infiniopDestroyTensorDescriptor(infiniopTensorDescriptor_t desc);

#endif // __INFINIOP_TENSOR_DESCRIPTOR__
