#ifndef __AVG_POOL1D_H__
#define __AVG_POOL1D_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"
#include "infiniop/ops/avg_pool1d.h"

#define DESCRIPTOR(NAMESPACE)                                    \
    namespace op::avg_pool1d::NAMESPACE {                        \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        AvgPool1dInfo _info;                                     \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            AvgPool1dInfo info,                                  \
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
            size_t kernel_size,                                  \
            size_t stride,                                       \
            size_t padding);                                     \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace,                                     \
            size_t workspace_size,                               \
            void *y,                                             \
            const void *x,                                       \
            void *stream) const;                                 \
    };                                                           \
    }

class AvgPool1dInfo {
private:
    AvgPool1dInfo() = default;

public:
    infiniDtype_t dtype;
    size_t batch, channels, in_width, out_width;
    size_t kernel_size, stride, padding;

    ptrdiff_t y_stride_batch, y_stride_channel, y_stride_width;
    ptrdiff_t x_stride_batch, x_stride_channel, x_stride_width;

    static utils::Result<AvgPool1dInfo> createAvgPool1dInfo(
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t x_desc,
        size_t kernel_size,
        size_t stride,
        size_t padding) {

        CHECK_OR_RETURN(y_desc != nullptr && x_desc != nullptr, INFINI_STATUS_NULL_POINTER);

        const infiniDtype_t dtype = y_desc->dtype();
        CHECK_OR_RETURN(dtype == x_desc->dtype(), INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);

        CHECK_OR_RETURN(y_desc->ndim() == 3 && x_desc->ndim() == 3, INFINI_STATUS_BAD_TENSOR_SHAPE);

        size_t batch = x_desc->dim(0);
        size_t channels = x_desc->dim(1);
        size_t in_width = x_desc->dim(2);

        CHECK_OR_RETURN(y_desc->dim(0) == batch, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(y_desc->dim(1) == channels, INFINI_STATUS_BAD_TENSOR_SHAPE);

        size_t padded_len = in_width + 2 * padding;

        CHECK_OR_RETURN(padded_len >= kernel_size, INFINI_STATUS_BAD_TENSOR_SHAPE);

        size_t expected_out_width = (padded_len - kernel_size) / stride + 1;
        CHECK_OR_RETURN(y_desc->dim(2) == expected_out_width, INFINI_STATUS_BAD_TENSOR_SHAPE);

        size_t out_width = expected_out_width;

        return utils::Result<AvgPool1dInfo>(AvgPool1dInfo{
            dtype,
            batch, channels, in_width, out_width,
            kernel_size, stride, padding,
            y_desc->stride(0), y_desc->stride(1), y_desc->stride(2),
            x_desc->stride(0), x_desc->stride(1), x_desc->stride(2)});
    }
};

#endif
