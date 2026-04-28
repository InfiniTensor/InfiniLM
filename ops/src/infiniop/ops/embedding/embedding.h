#ifndef __EMBEDDING_H__
#define __EMBEDDING_H__

#include "../../../utils.h"
#include "../../operator.h"

#define DESCRIPTOR(NAMESPACE)                             \
                                                          \
    namespace op::embedding::NAMESPACE {                  \
    class Descriptor final : public InfiniopDescriptor {  \
        struct Opaque;                                    \
        Opaque *_opaque;                                  \
        size_t _num_indices;                              \
        size_t _embedding_dim;                            \
        size_t _vocab_size;                               \
        infiniDtype_t _input_dtype;                       \
        infiniDtype_t _weight_dtype;                      \
                                                          \
        Descriptor(                                       \
            size_t num_indices,                           \
            size_t embedding_dim,                         \
            size_t vocab_size,                            \
            infiniDtype_t input_dtype,                    \
            infiniDtype_t weight_dtype,                   \
            Opaque *opaque,                               \
            infiniDevice_t device_type,                   \
            int device_id)                                \
            : InfiniopDescriptor{device_type, device_id}, \
              _opaque(opaque),                            \
              _num_indices(num_indices),                  \
              _embedding_dim(embedding_dim),              \
              _vocab_size(vocab_size),                    \
              _input_dtype(input_dtype),                  \
              _weight_dtype(weight_dtype) {}              \
                                                          \
    public:                                               \
        ~Descriptor();                                    \
                                                          \
        static infiniStatus_t create(                     \
            infiniopHandle_t handle,                      \
            Descriptor **desc_ptr,                        \
            infiniopTensorDescriptor_t output_desc,       \
            infiniopTensorDescriptor_t input_desc,        \
            infiniopTensorDescriptor_t weight_desc);      \
                                                          \
        infiniStatus_t calculate(                         \
            void *output,                                 \
            const void *input,                            \
            const void *weight,                           \
            void *stream) const;                          \
    };                                                    \
    }

#endif // __EMBEDDING_H__
