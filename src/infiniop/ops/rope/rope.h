#ifndef __ROPE_H__
#define __ROPE_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::rope::NAMESPACE {                              \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        RoPEInfo _info;                                          \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            RoPEInfo info,                                       \
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
            infiniopTensorDescriptor_t pos_desc,                 \
            infiniopTensorDescriptor_t sin_desc,                 \
            infiniopTensorDescriptor_t cos_desc);                \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace,                                     \
            size_t workspace_size,                               \
            void *y,                                             \
            const void *x,                                       \
            const void *pos_ids,                                 \
            const void *sin_table,                               \
            const void *cos_table,                               \
            void *stream) const;                                 \
    };                                                           \
    }

class RoPEInfo {
private:
    RoPEInfo() = default;

public:
    infiniDtype_t data_type, pos_type;
    size_t seqlen, nhead, dhead, table_len, table_dim;
    ptrdiff_t
        y_stride_seqlen,
        y_stride_nhead,
        x_stride_seqlen,
        x_stride_nhead;

    static utils::Result<RoPEInfo> createRoPEInfo(
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t pos_desc,
        infiniopTensorDescriptor_t sin_desc,
        infiniopTensorDescriptor_t cos_desc) {
        CHECK_OR_RETURN(
            y_desc != nullptr && pos_desc != nullptr && sin_desc != nullptr && cos_desc != nullptr,
            INFINI_STATUS_NULL_POINTER);

        const infiniDtype_t data_type = y_desc->dtype();
        const infiniDtype_t pos_type = pos_desc->dtype();
        CHECK_OR_RETURN(data_type == x_desc->dtype() && data_type == sin_desc->dtype() && data_type == cos_desc->dtype(),
                        INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_DTYPE(data_type, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);
        CHECK_DTYPE_ANY_INT(pos_type);

        CHECK_OR_RETURN(y_desc->ndim() == 3
                            && x_desc->ndim() == 3
                            && pos_desc->ndim() == 1
                            && sin_desc->ndim() == 2
                            && cos_desc->ndim() == 2,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);

        const auto seqlen = y_desc->dim(0),
                   nhead = y_desc->dim(1),
                   dhead = y_desc->dim(2),
                   table_len = sin_desc->dim(0),
                   table_dim = sin_desc->dim(1);

        CHECK_OR_RETURN(seqlen == x_desc->dim(0)
                            && seqlen == pos_desc->dim(0)
                            && nhead == x_desc->dim(1) && dhead == x_desc->dim(2)
                            && table_len == cos_desc->dim(0) && table_dim == cos_desc->dim(1),
                        INFINI_STATUS_BAD_TENSOR_SHAPE);

        CHECK_OR_RETURN(dhead == table_dim * 2, INFINI_STATUS_BAD_TENSOR_SHAPE);
        // Last dimension of x and y must be contiguous
        CHECK_OR_RETURN(y_desc->stride(2) == 1 && x_desc->stride(2) == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);
        // sin table and cos table must be totally contiguous
        CHECK_OR_RETURN(sin_desc->stride(1) == 1
                            && cos_desc->stride(1) == 1
                            && sin_desc->stride(0) == ptrdiff_t(table_dim)
                            && cos_desc->stride(0) == ptrdiff_t(table_dim),
                        INFINI_STATUS_BAD_TENSOR_STRIDES);

        return utils::Result<RoPEInfo>(RoPEInfo{
            data_type,
            pos_type,
            seqlen,
            nhead,
            dhead,
            table_len,
            table_dim,
            y_desc->stride(0),
            y_desc->stride(1),
            x_desc->stride(0),
            x_desc->stride(1),
        });
    }
};

#endif
