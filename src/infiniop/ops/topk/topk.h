#ifndef __TOPK_H__
#define __TOPK_H__

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                                  \
                                                                               \
    namespace op::topk::NAMESPACE {                                            \
    class Descriptor final : public InfiniopDescriptor {                       \
        struct Opaque;                                                         \
        Opaque *_opaque;                                                       \
        TopKInfo _info;                                                        \
                                                                               \
        Descriptor(                                                            \
            TopKInfo info,                                                     \
            Opaque *opaque,                                                    \
            infiniDevice_t device_type,                                        \
            int device_id)                                                     \
            : InfiniopDescriptor{device_type, device_id},                      \
              _opaque(opaque),                                                 \
              _info(info) {}                                                   \
                                                                               \
    public:                                                                    \
        ~Descriptor();                                                         \
                                                                               \
        static infiniStatus_t create(                                          \
            infiniopHandle_t handle,                                           \
            Descriptor **desc_ptr,                                             \
            infiniopTensorDescriptor_t input_desc,                             \
            infiniopTensorDescriptor_t output_val_desc,                        \
            infiniopTensorDescriptor_t output_ind_desc, int k);                \
                                                                               \
        infiniStatus_t calculate(const void *input, void *output_val,          \
                                 void *output_ind, void *workspace,            \
                                 void *stream) const;                          \
                                                                               \
        size_t getWorkspaceSize() const;                                       \
    };                                                                         \
    }

#endif // __TOPK_H__ 