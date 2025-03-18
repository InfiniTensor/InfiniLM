#ifndef CAUSAL_SOFTMAX_H
#define CAUSAL_SOFTMAX_H

#include "../../operator.h"
#include "../../tensor.h"
#include <iostream>
#include <vector>

struct CausalSoftmaxInfo {
    infiniDtype_t dtype;
    size_t batch_size;
    ptrdiff_t stride_b;
    size_t seq_len;
    ptrdiff_t stride_i;
    size_t total_seq_len;
    ptrdiff_t stride_j;
};

inline infiniStatus_t createCausalSoftmaxInfo(CausalSoftmaxInfo *info, infiniopTensorDescriptor_t y_desc) {
    auto dtype = y_desc->dtype();
    if (y_desc->dtype() != INFINI_DTYPE_F16 && y_desc->dtype() != INFINI_DTYPE_F32) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    info->dtype = dtype;

    if (y_desc->ndim() != 2 && y_desc->ndim() != 3) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    if (y_desc->shape()[y_desc->ndim() - 1] < y_desc->shape()[y_desc->ndim() - 2]) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    size_t batch_size = 1;
    ptrdiff_t stride_b = 0;
    size_t seq_len = y_desc->shape()[y_desc->ndim() - 2];
    ptrdiff_t stride_i = y_desc->strides()[y_desc->ndim() - 2];
    size_t total_seq_len = y_desc->shape()[y_desc->ndim() - 1];
    ptrdiff_t stride_j = y_desc->strides()[y_desc->ndim() - 1];
    if (y_desc->ndim() == 3) {
        stride_b = y_desc->strides()[0];
        batch_size = y_desc->shape()[0];
    }

    info->batch_size = batch_size;
    info->stride_b = stride_b;
    info->seq_len = seq_len;
    info->stride_i = stride_i;
    info->total_seq_len = total_seq_len;
    info->stride_j = stride_j;

    return INFINI_STATUS_SUCCESS;
}

#define DESCRIPTOR(NAMESPACE)                                            \
    namespace op::causal_softmax::NAMESPACE {                            \
    class Descriptor final : public InfiniopDescriptor {                 \
        struct Opaque;                                                   \
        Opaque *_opaque;                                                 \
        CausalSoftmaxInfo _info;                                         \
        size_t _workspace_size;                                          \
                                                                         \
        Descriptor(                                                      \
            Opaque *opaque,                                              \
            CausalSoftmaxInfo info,                                      \
            size_t workspace_size,                                       \
            infiniDevice_t device_type,                                  \
            int device_id) : InfiniopDescriptor{device_type, device_id}, \
                             _opaque(opaque),                            \
                             _info(info),                                \
                             _workspace_size(workspace_size) {}          \
                                                                         \
    public:                                                              \
        ~Descriptor();                                                   \
        size_t workspaceSize() const { return _workspace_size; }         \
        static infiniStatus_t create(                                    \
            infiniopHandle_t handle,                                     \
            Descriptor **desc_ptr,                                       \
            infiniopTensorDescriptor_t y_desc);                          \
        infiniStatus_t calculate(void *workspace, size_t workspace_size, \
                                 void *data, void *stream);              \
    };                                                                   \
    }

#endif // CAUSAL_SOFTMAX_H
