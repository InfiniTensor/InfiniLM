#ifndef __ROPE_H__
#define __ROPE_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"
#include "infiniop/ops/rope.h"

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
            infiniopTensorDescriptor_t cos_desc,                 \
            infiniopRoPEAlgo_t algo);                            \
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
    size_t batch, seqlen, nhead, dhead, table_len, table_dim;
    ptrdiff_t
        y_stride_batch, // Batch stride (0 for 3D tensors)
        y_stride_seqlen,
        y_stride_nhead,
        x_stride_batch, // Batch stride (0 for 3D tensors)
        x_stride_seqlen,
        x_stride_nhead;
    bool has_batch_dim;     // Whether tensors have batch dimension
    bool pos_has_batch_dim; // Whether position IDs have batch dimension
    infiniopRoPEAlgo_t algo;

    static utils::Result<RoPEInfo>
    createRoPEInfo(
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t pos_desc,
        infiniopTensorDescriptor_t sin_desc,
        infiniopTensorDescriptor_t cos_desc,
        infiniopRoPEAlgo_t algo) {
        CHECK_OR_RETURN(
            y_desc != nullptr && pos_desc != nullptr && sin_desc != nullptr && cos_desc != nullptr && algo < infiniopRoPEAlgo_t::INFINIOP_ROPE_ALGO_COUNT,
            INFINI_STATUS_NULL_POINTER);

        const infiniDtype_t data_type = y_desc->dtype();
        const infiniDtype_t pos_type = pos_desc->dtype();
        CHECK_OR_RETURN(data_type == x_desc->dtype() && data_type == sin_desc->dtype() && data_type == cos_desc->dtype(),
                        INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_DTYPE(data_type, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);
        CHECK_DTYPE_ANY_INT(pos_type);

        // Support both 3D (no batch) and 4D (with batch) tensors
        bool y_has_batch = y_desc->ndim() == 4;
        bool x_has_batch = x_desc->ndim() == 4;

        CHECK_OR_RETURN(y_has_batch == x_has_batch, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN((y_has_batch && x_has_batch) || (y_desc->ndim() == 3 && x_desc->ndim() == 3), INFINI_STATUS_BAD_TENSOR_SHAPE);

        // Check position IDs: can be 1D (shared) or 2D (per-batch)
        bool pos_has_batch = pos_desc->ndim() == 2;
        CHECK_OR_RETURN(pos_desc->ndim() == 1 || pos_desc->ndim() == 2,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);

        CHECK_OR_RETURN(sin_desc->ndim() == 2 && cos_desc->ndim() == 2,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);

        size_t batch, seqlen, nhead, dhead;
        if (y_has_batch) {
            // 4D tensors: [batch, seqlen, nhead, dhead]
            batch = y_desc->dim(0);
            seqlen = y_desc->dim(1);
            nhead = y_desc->dim(2);
            dhead = y_desc->dim(3);

            CHECK_OR_RETURN(batch == x_desc->dim(0) && seqlen == x_desc->dim(1) && nhead == x_desc->dim(2) && dhead == x_desc->dim(3),
                            INFINI_STATUS_BAD_TENSOR_SHAPE);
        } else {
            // 3D tensors: [seqlen, nhead, dhead] (batch = 1)
            batch = 1;
            seqlen = y_desc->dim(0);
            nhead = y_desc->dim(1);
            dhead = y_desc->dim(2);

            CHECK_OR_RETURN(seqlen == x_desc->dim(0) && nhead == x_desc->dim(1) && dhead == x_desc->dim(2),
                            INFINI_STATUS_BAD_TENSOR_SHAPE);
        }

        const auto table_len = sin_desc->dim(0);
        const auto table_dim = sin_desc->dim(1);

        // Check position IDs shape
        if (pos_has_batch) {
            // 2D position IDs: [batch, seqlen] or [batch, seqlen?]
            CHECK_OR_RETURN(batch == pos_desc->dim(0) && seqlen == pos_desc->dim(1),
                            INFINI_STATUS_BAD_TENSOR_SHAPE);
        } else {
            // 1D position IDs: [seqlen]
            CHECK_OR_RETURN(seqlen == pos_desc->dim(0),
                            INFINI_STATUS_BAD_TENSOR_SHAPE);
        }

        CHECK_OR_RETURN(table_len == cos_desc->dim(0) && table_dim == cos_desc->dim(1),
                        INFINI_STATUS_BAD_TENSOR_SHAPE);

        CHECK_OR_RETURN(dhead == table_dim * 2, INFINI_STATUS_BAD_TENSOR_SHAPE);

        // Last dimension of x and y must be contiguous
        if (y_has_batch) {
            CHECK_OR_RETURN(y_desc->stride(3) == 1 && x_desc->stride(3) == 1,
                            INFINI_STATUS_BAD_TENSOR_STRIDES);
        } else {
            CHECK_OR_RETURN(y_desc->stride(2) == 1 && x_desc->stride(2) == 1,
                            INFINI_STATUS_BAD_TENSOR_STRIDES);
        }

        // sin table and cos table must be totally contiguous
        CHECK_OR_RETURN(sin_desc->isContiguous() && cos_desc->isContiguous(),
                        INFINI_STATUS_BAD_TENSOR_STRIDES);

        // Set strides based on tensor dimensions
        ptrdiff_t y_stride_batch, y_stride_seqlen, y_stride_nhead;
        ptrdiff_t x_stride_batch, x_stride_seqlen, x_stride_nhead;

        if (y_has_batch) {
            y_stride_batch = y_desc->stride(0);
            y_stride_seqlen = y_desc->stride(1);
            y_stride_nhead = y_desc->stride(2);

            x_stride_batch = x_desc->stride(0);
            x_stride_seqlen = x_desc->stride(1);
            x_stride_nhead = x_desc->stride(2);
        } else {
            // For 3D tensors, set batch stride to 0 (no batch dimension)
            y_stride_batch = 0;
            y_stride_seqlen = y_desc->stride(0);
            y_stride_nhead = y_desc->stride(1);

            x_stride_batch = 0;
            x_stride_seqlen = x_desc->stride(0);
            x_stride_nhead = x_desc->stride(1);
        }

        return utils::Result<RoPEInfo>(RoPEInfo{
            data_type,
            pos_type,
            batch,
            seqlen,
            nhead,
            dhead,
            table_len,
            table_dim,
            y_stride_batch,
            y_stride_seqlen,
            y_stride_nhead,
            x_stride_batch,
            x_stride_seqlen,
            x_stride_nhead,
            y_has_batch,   // has_batch_dim
            pos_has_batch, // pos_has_batch_dim
            algo,
        });
    }
};

#endif
