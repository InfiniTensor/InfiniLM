#ifndef INFINIOP_VAR_DESCRIPTOR_H_
#define INFINIOP_VAR_DESCRIPTOR_H_
#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"

#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::var::NAMESPACE {                               \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        VarInfo _info;                                           \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            Opaque *opaque,                                      \
            VarInfo info,                                        \
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
        size_t workspaceSize() const { return _workspace_size; } \
                                                                 \
        static infiniStatus_t create(                            \
            infiniopHandle_t handle,                             \
            Descriptor **desc_ptr,                               \
            infiniopTensorDescriptor_t var_output_desc,          \
            infiniopTensorDescriptor_t input_desc,               \
            size_t *dim,                                         \
            size_t dim_size,                                     \
            bool unbiased,                                       \
            bool keepdim);                                       \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace, size_t workspace_size,              \
            void *var_output,                                    \
            const void *input,                                   \
            bool unbiased,                                       \
            bool keepdim,                                        \
            void *stream) const;                                 \
    };                                                           \
    }

#endif
