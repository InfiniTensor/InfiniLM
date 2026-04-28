#ifndef ADD_RMS_NORM_H
#define ADD_RMS_NORM_H

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::add_rms_norm::NAMESPACE {                      \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        AddRMSNormInfo _info;                                    \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            Opaque *opaque,                                      \
            AddRMSNormInfo info,                                 \
            size_t workspace_size,                               \
            infiniDevice_t device_type,                          \
            int device_id)                                       \
            : InfiniopDescriptor{device_type, device_id},        \
              _opaque(opaque),                                   \
              _info(info),                                       \
              _workspace_size(workspace_size) {}                 \
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
            infiniopTensorDescriptor_t residual_out_desc,        \
            infiniopTensorDescriptor_t a_desc,                   \
            infiniopTensorDescriptor_t b_desc,                   \
            infiniopTensorDescriptor_t weight_desc,              \
            float epsilon);                                      \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace, size_t workspace_size,              \
            void *y,                                             \
            void *residual_out,                                  \
            const void *a,                                       \
            const void *b,                                       \
            const void *weight,                                  \
            void *stream) const;                                 \
    };                                                           \
    }

#endif // ADD_RMS_NORM_H
