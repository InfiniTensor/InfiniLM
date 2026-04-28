#ifndef INFINIOP_ADDR_DESCRIPTOR_H_
#define INFINIOP_ADDR_DESCRIPTOR_H_
#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"
#include "infiniop/ops/addr.h"
#include <cstddef>

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::addr::NAMESPACE {                              \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        AddrInfo _info;                                          \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            AddrInfo info,                                       \
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
            infiniopTensorDescriptor_t out_desc,                 \
            infiniopTensorDescriptor_t input_desc,               \
            infiniopTensorDescriptor_t vec1_desc,                \
            infiniopTensorDescriptor_t vec2_desc,                \
            float beta,                                          \
            float alpha);                                        \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace,                                     \
            size_t workspace_size,                               \
            void *out,                                           \
            const void *input,                                   \
            const void *vec1,                                    \
            const void *vec2,                                    \
            void *stream) const;                                 \
    };                                                           \
    }

struct AddrInfo {
    infiniDtype_t dtype;
    size_t vec1_size;
    size_t vec2_size;
    float beta;
    float alpha;
    ptrdiff_t input_stride0, input_stride1;
    ptrdiff_t output_stride0, output_stride1;
    ptrdiff_t vec1_stride;
    ptrdiff_t vec2_stride;

    static utils::Result<AddrInfo>
    create(infiniopTensorDescriptor_t input_desc,
           infiniopTensorDescriptor_t output_desc,
           infiniopTensorDescriptor_t vec1_desc,
           infiniopTensorDescriptor_t vec2_desc,
           float beta = 1.0f, float alpha = 1.0f) {
        CHECK_OR_RETURN(input_desc->ndim() == 2, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(vec1_desc->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(vec2_desc->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(input_desc->dim(0) == vec1_desc->dim(0) && input_desc->dim(1) == vec2_desc->dim(0), INFINI_STATUS_BAD_TENSOR_SHAPE);
        const infiniDtype_t data_type = input_desc->dtype();
        CHECK_DTYPE(data_type, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);

        return utils::Result<AddrInfo>(AddrInfo{
            data_type,
            vec1_desc->dim(0),
            vec2_desc->dim(0),
            beta,
            alpha,
            input_desc->stride(0),
            input_desc->stride(1),
            output_desc->stride(0),
            output_desc->stride(1),
            vec1_desc->stride(0),
            vec2_desc->stride(0),
        });
    }
};

#endif
