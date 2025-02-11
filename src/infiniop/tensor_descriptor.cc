#include "infiniop/tensor_descriptor.h"
#include <cstring>

__C __export infiniopStatus_t infiniopCreateTensorDescriptor(infiniopTensorDescriptor_t *desc_ptr, size_t ndim, size_t const *shape_, int64_t const *strides_, infiniDtype_t datatype) {
    size_t *shape = new size_t[ndim];
    int64_t *strides = new int64_t[ndim];
    std::memcpy(shape, shape_, ndim * sizeof(size_t));
    if (strides_) {
        std::memcpy(strides, strides_, ndim * sizeof(int64_t));
    } else {
        int64_t dsize = 1;
        for (int i = ndim - 1; i >= 0; i--) {
            strides[i] = dsize;
            dsize *= shape[i];
        }
    }
    *desc_ptr = new InfiniopTensorDescriptor{datatype, ndim, shape, strides};
    return INFINIOP_STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDestroyTensorDescriptor(infiniopTensorDescriptor_t desc) {
    delete[] desc->shape;
    delete[] desc->strides;
    delete desc;
    return INFINIOP_STATUS_SUCCESS;
}
