#ifndef INFINIOP_TOPK_DESCRIPTOR_H_
#define INFINIOP_TOPK_DESCRIPTOR_H_
#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"

#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::topk::NAMESPACE {                              \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        TopKInfo _info;                                          \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            Opaque *opaque,                                      \
            TopKInfo info,                                       \
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
            infiniopTensorDescriptor_t values_output_desc,       \
            infiniopTensorDescriptor_t indices_output_desc,      \
            infiniopTensorDescriptor_t input_desc,               \
            size_t k,                                            \
            size_t dim,                                          \
            bool largest,                                        \
            bool sorted);                                        \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace, size_t workspace_size,              \
            void *values_output,                                 \
            void *indices_output,                                \
            const void *input,                                   \
            size_t k,                                            \
            size_t dim,                                          \
            bool largest,                                        \
            bool sorted,                                         \
            void *stream) const;                                 \
    };                                                           \
    }

#endif
