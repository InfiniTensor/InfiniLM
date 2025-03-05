#ifndef __COMMON_BANG_H__
#define __COMMON_BANG_H__

#include "../../../utils.h"
#include "../pool.h"
#include "bang_handle.h"
#include "cnnl.h"
#include "cnrt.h"
#include "infiniop/tensor_descriptor.h"
#include <memory>
#include <vector>

// the maximum NRAM memory is 1024 * 768
#define NRAM_MAX_SIZE (1024 * 256)

#define GDRAM_MAX_SIZE (1024 * 1024 * 1024)

struct InfiniopBangHandle {
    infiniDevice_t device;
    int device_id;
    std::shared_ptr<Pool<cnnlHandle_t>> cnnl_handle_pool;
};

inline cnnlDataType_t cnnlDataTypeConvert(infiniDtype_t dataType) {
    switch (dataType) {
    case INFINI_DTYPE_F32:
        return CNNL_DTYPE_FLOAT;
    case INFINI_DTYPE_F64:
        return CNNL_DTYPE_DOUBLE;
    case INFINI_DTYPE_F16:
        return CNNL_DTYPE_HALF;
    case INFINI_DTYPE_I8:
        return CNNL_DTYPE_INT8;
    case INFINI_DTYPE_I32:
        return CNNL_DTYPE_INT32;
    case INFINI_DTYPE_U8:
        return CNNL_DTYPE_UINT8;
    case INFINI_DTYPE_BF16:
        return CNNL_DTYPE_BFLOAT16;
    case INFINI_DTYPE_I64:
        return CNNL_DTYPE_INT64;
    default:
        return CNNL_DTYPE_INVALID;
    }
}

template <typename T>
void use_cnnl(std::shared_ptr<Pool<cnnlHandle_t>> &pool, cnrtQueue_t queue,
              T const &f) {
    auto handle = pool->pop();
    if (!handle) {
        cnnlCreate(&(*handle));
    }
    cnnlSetQueue(*handle, (cnrtQueue_t)queue);
    f(*handle);
    pool->push(std::move(*handle));
}

template <typename T>
void use_cnnl(std::shared_ptr<Pool<cnnlHandle_t>> &pool, T const &f) {
    auto handle = pool->pop();
    if (!handle) {
        cnnlCreate(&(*handle));
    }
    f(*handle);
    pool->push(std::move(*handle));
}

// set cnnl tensor descriptor without strides11
inline void setCnnlTensor(cnnlTensorDescriptor_t desc,
                          const infiniopTensorDescriptor_t layout) {
    std::vector<int> dims(layout->ndim);
    for (size_t i = 0; i < layout->ndim; i++) {
        dims[i] = static_cast<int>(layout->shape[i]);
    }
    cnnlSetTensorDescriptor(desc, CNNL_LAYOUT_ARRAY,
                            cnnlDataTypeConvert(layout->dtype), dims.size(),
                            dims.data());
}

// set cnnl tensor descriptor with strides
inline void setCnnlTensorEx(cnnlTensorDescriptor_t desc,
                            const infiniopTensorDescriptor_t layout) {
    std::vector<int> dim_size(layout->ndim), dim_stride(layout->ndim);
    for (size_t i = 0; i < layout->ndim; i++) {
        dim_size[i] = static_cast<int>(layout->shape[i]);
        dim_stride[i] = static_cast<int>(layout->strides[i]);
    }
    cnnlSetTensorDescriptorEx(
        desc, CNNL_LAYOUT_ARRAY, cnnlDataTypeConvert(layout->dtype),
        dim_size.size(), dim_size.data(), dim_stride.data());
}

#endif // __COMMON_BANG_H__
