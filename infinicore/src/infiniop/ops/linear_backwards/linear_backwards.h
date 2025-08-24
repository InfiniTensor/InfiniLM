#ifndef __LINEAR_BACKWARDS_H__
#define __LINEAR_BACKWARDS_H__

#include "../../operator.h"
#include "info.h"

/**
 * # Linear Backwards Operator Descriptor Macro
 * 
 * The linear backwards operator computes gradients for backpropagation:
 * - grad_input = grad_output @ weight
 * - grad_weight = grad_output.T @ input
 * - grad_bias = sum(grad_output, dim=0) [optional]
 */

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::linear_backwards::NAMESPACE {                  \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        infiniDtype_t _dtype;                                    \
        LinearBackwardsInfo _info;                               \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            infiniDtype_t dtype,                                 \
            LinearBackwardsInfo info,                            \
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
            infiniopTensorDescriptor_t grad_input_desc,          \
            infiniopTensorDescriptor_t grad_weight_desc,         \
            infiniopTensorDescriptor_t grad_bias_desc,           \
            infiniopTensorDescriptor_t grad_output_desc,         \
            infiniopTensorDescriptor_t input_desc,               \
            infiniopTensorDescriptor_t weight_desc);             \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace, size_t workspace_size,              \
            void *grad_input,                                    \
            void *grad_weight,                                   \
            void *grad_bias,                                     \
            const void *grad_output,                             \
            const void *input,                                   \
            const void *weight,                                  \
            void *stream) const;                                 \
    };                                                           \
    }

#endif // __LINEAR_BACKWARDS_H__