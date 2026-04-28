#ifndef INFINIOP_ADAPTIVE_AVG_POOL3D_DESCRIPTOR_H_
#define INFINIOP_ADAPTIVE_AVG_POOL3D_DESCRIPTOR_H_
#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"
#include "infiniop/ops/adaptive_avg_pool3d.h"
#include <cstddef>

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::adaptive_avg_pool3d::NAMESPACE {               \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        AdaptiveAvgPool3DInfo _info;                             \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            AdaptiveAvgPool3DInfo info,                          \
            size_t workspace_size_,                              \
            Opaque *opaque,                                      \
            infiniDevice_t device_type,                          \
            int device_id)                                       \
            : InfiniopDescriptor{device_type, device_id},        \
              _opaque(opaque),                                   \
              _info(info),                                       \
              _workspace_size(workspace_size_) {}                \
                                                                 \
    public:                                                      \
        ~Descriptor();                                           \
                                                                 \
        size_t workspaceSize() const { return _workspace_size; } \
                                                                 \
        static infiniStatus_t create(                            \
            infiniopHandle_t handle,                             \
            Descriptor **desc_ptr,                               \
            infiniopTensorDescriptor_t y_desc,                   \
            infiniopTensorDescriptor_t x_desc,                   \
            size_t *output_size);                                \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace,                                     \
            size_t workspace_size,                               \
            void *y,                                             \
            const void *x,                                       \
            void *stream) const;                                 \
    };                                                           \
    }

class AdaptiveAvgPool3DInfo {
private:
    AdaptiveAvgPool3DInfo() = default;

public:
    infiniDtype_t dtype;
    size_t x_d, x_h, x_w;
    size_t y_d, y_h, y_w;
    size_t N, C;
    std::vector<ptrdiff_t> x_strides;
    std::vector<ptrdiff_t> y_strides;

    static utils::Result<AdaptiveAvgPool3DInfo>
    create(
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t x_desc,
        size_t *output_size) {
        CHECK_OR_RETURN(x_desc != nullptr && output_size != nullptr,
                        INFINI_STATUS_NULL_POINTER);

        const infiniDtype_t data_type = x_desc->dtype();
        CHECK_DTYPE(data_type, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);
        const size_t ndim = x_desc->ndim();
        CHECK_OR_RETURN(ndim == 5, INFINI_STATUS_BAD_TENSOR_SHAPE);

        return utils::Result<AdaptiveAvgPool3DInfo>(AdaptiveAvgPool3DInfo{
            data_type,
            x_desc->dim(2),
            x_desc->dim(3),
            x_desc->dim(4),
            output_size[0],
            output_size[1],
            output_size[2],
            x_desc->dim(0),
            x_desc->dim(1),
            x_desc->strides(),
            y_desc->strides()});
    }
};

#endif
