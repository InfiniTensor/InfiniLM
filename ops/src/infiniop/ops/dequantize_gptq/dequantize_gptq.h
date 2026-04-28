#ifndef __DEQUANTIZE_GPTQ_H__
#define __DEQUANTIZE_GPTQ_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::dequantize_gptq::NAMESPACE {                   \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        DequantizeGPTQInfo _info;                                \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            size_t workspace_size_,                              \
            Opaque *opaque,                                      \
            DequantizeGPTQInfo info,                             \
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
            infiniopTensorDescriptor_t zeros_desc,               \
            infiniopTensorDescriptor_t g_idx_desc);              \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace,                                     \
            size_t workspace_size,                               \
            void *out,                                           \
            const void *qweight,                                 \
            const void *scales,                                  \
            const void *zeros,                                   \
            const void *g_idx,                                   \
            void *stream) const;                                 \
    };                                                           \
    }

#endif //__DEQUANTIZE_GPTQ_H__
