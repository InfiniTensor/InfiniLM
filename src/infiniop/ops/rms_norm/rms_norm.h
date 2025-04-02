#ifndef RMS_NORM_H
#define RMS_NORM_H
#include "../../operator.h"
#include "../../tensor.h"
#include <vector>

struct RMSNormInfo {
    infiniDtype_t wtype;
    infiniDtype_t atype;
    float epsilon;
    std::vector<size_t> shape;
    std::vector<ptrdiff_t> y_strides;
    std::vector<ptrdiff_t> x_strides;

    size_t ndim() { return shape.size(); }
    size_t dim() { return shape[ndim() - 1]; }
};

inline infiniStatus_t createRMSNormInfo(RMSNormInfo *info, infiniopTensorDescriptor_t y_desc,
                                        infiniopTensorDescriptor_t x_desc,
                                        infiniopTensorDescriptor_t w_desc,
                                        float epsilon) {
    auto atype = y_desc->dtype();
    auto wtype = w_desc->dtype();
    if (x_desc->dtype() != atype) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    if (atype == INFINI_DTYPE_F16) {
        if (wtype != INFINI_DTYPE_F16 && wtype != INFINI_DTYPE_F32) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    } else if (atype == INFINI_DTYPE_F32 || atype == INFINI_DTYPE_F64) {
        if (atype != wtype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    info->wtype = wtype;
    info->atype = atype;

    info->epsilon = epsilon;

    if (y_desc->ndim() != 2 || x_desc->ndim() != 2 || w_desc->ndim() != 1) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    size_t batch = y_desc->shape()[0];
    size_t dim = y_desc->shape()[1];
    if (x_desc->shape()[0] != batch || x_desc->shape()[1] != dim || w_desc->shape()[0] != dim) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    if (w_desc->stride(0) != 1) {
        return INFINI_STATUS_BAD_TENSOR_STRIDES;
    }

    if (x_desc->stride(1) != 1 || y_desc->stride(1) != 1) {
        return INFINI_STATUS_BAD_TENSOR_STRIDES;
    }

    info->shape = std::move(y_desc->shape());
    info->y_strides = std::move(y_desc->strides());
    info->x_strides = std::move(x_desc->strides());

    return INFINI_STATUS_SUCCESS;
}

#define DESCRIPTOR(NAMESPACE)                                                          \
    namespace op::rms_norm::NAMESPACE {                                                \
    class Descriptor final : public InfiniopDescriptor {                               \
        struct Opaque;                                                                 \
        Opaque *_opaque;                                                               \
        RMSNormInfo _info;                                                             \
        size_t _workspace_size;                                                        \
                                                                                       \
        Descriptor(                                                                    \
            Opaque *opaque,                                                            \
            RMSNormInfo info,                                                          \
            size_t workspace_size,                                                     \
            infiniDevice_t device_type,                                                \
            int device_id) : InfiniopDescriptor{device_type, device_id},               \
                             _opaque(opaque),                                          \
                             _info(info),                                              \
                             _workspace_size(workspace_size) {}                        \
                                                                                       \
    public:                                                                            \
        ~Descriptor();                                                                 \
        size_t workspaceSize() const { return _workspace_size; }                       \
        static infiniStatus_t create(                                                  \
            infiniopHandle_t handle,                                                   \
            Descriptor **desc_ptr,                                                     \
            infiniopTensorDescriptor_t y_desc,                                         \
            infiniopTensorDescriptor_t x_desc,                                         \
            infiniopTensorDescriptor_t w_desc,                                         \
            float epsilon);                                                            \
        infiniStatus_t calculate(void *workspace, size_t workspace_size,               \
                                 void *y, const void *x, const void *w, void *stream); \
    };                                                                                 \
    }

#endif // RMS_NORM_H
