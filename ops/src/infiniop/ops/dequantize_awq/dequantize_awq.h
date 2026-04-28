#ifndef __DEQUANTIZE_AWQ_H__
#define __DEQUANTIZE_AWQ_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::dequantize_awq::NAMESPACE {                    \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        DequantizeAWQInfo _info;                                 \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            size_t workspace_size_,                              \
            Opaque *opaque,                                      \
            DequantizeAWQInfo info,                              \
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
            infiniopTensorDescriptor_t qweight_desc,             \
            infiniopTensorDescriptor_t scales_desc,              \
            infiniopTensorDescriptor_t zeros_desc);              \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace,                                     \
            size_t workspace_size,                               \
            void *out,                                           \
            const void *qweight,                                 \
            const void *scales,                                  \
            const void *zeros,                                   \
            void *stream) const;                                 \
    };                                                           \
    }

#endif //__DEQUANTIZE_AWQ_H__
