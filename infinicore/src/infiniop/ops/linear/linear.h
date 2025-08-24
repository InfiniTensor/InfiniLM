#ifndef __LINEAR_H__
#define __LINEAR_H__

#include "../../operator.h"
#include "info.h"

/**
 * # Linear Operator Descriptor Macro
 * 
 * The linear operator performs: output = input @ weight.T + bias
 * where:
 * - input: shape (..., in_features)
 * - weight: shape (out_features, in_features) 
 * - bias: shape (out_features) [optional]
 * - output: shape (..., out_features)
 */

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::linear::NAMESPACE {                            \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        infiniDtype_t _dtype;                                    \
        LinearInfo _info;                                        \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            infiniDtype_t dtype,                                 \
            LinearInfo info,                                     \
            size_t workspace_size_,                              \
            Opaque *opaque,                                      \
            infiniDevice_t device_type,                          \
            int device_id)                                       \
            : InfiniopDescriptor{device_type, device_id},        \
              _opaque(opaque),                                   \
              _dtype(dtype),                                     \
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
            infiniopTensorDescriptor_t output_desc,              \
            infiniopTensorDescriptor_t input_desc,               \
            infiniopTensorDescriptor_t weight_desc,              \
            infiniopTensorDescriptor_t bias_desc);               \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace, size_t workspace_size,              \
            void *output,                                        \
            const void *input,                                   \
            const void *weight,                                  \
            const void *bias,                                    \
            void *stream) const;                                 \
    };                                                           \
    }

#endif // __LINEAR_H__