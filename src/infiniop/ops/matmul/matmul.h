#ifndef __MATMUL_H__
#define __MATMUL_H__

#include "blas.h"
#include "infiniop/handle.h"
#include "infiniop/operator.h"

#define DESCRIPTOR(NAMESPACE)                             \
                                                          \
    namespace matmul::NAMESPACE {                         \
    class Descriptor final : public InfiniopDescriptor {  \
        struct Opaque;                                    \
        Opaque *_opaque;                                  \
        infiniDtype_t _dtype;                             \
        MatmulInfo _info;                                 \
                                                          \
        Descriptor(                                       \
            infiniDtype_t dtype,                          \
            MatmulInfo info,                              \
            size_t workspace_size_,                       \
            Opaque *opaque,                               \
            infiniDevice_t device_type,                   \
            int device_id)                                \
            : InfiniopDescriptor{device_type, device_id}, \
              _opaque(opaque),                            \
              _dtype(dtype),                              \
              _info(info),                                \
              workspace_size(workspace_size_) {}          \
                                                          \
    public:                                               \
        size_t workspace_size;                            \
                                                          \
        ~Descriptor();                                    \
                                                          \
        static infiniopStatus_t create(                   \
            infiniopHandle_t handle,                      \
            Descriptor **desc_ptr,                        \
            infiniopTensorDescriptor_t c_desc,            \
            infiniopTensorDescriptor_t a_desc,            \
            infiniopTensorDescriptor_t b_desc);           \
                                                          \
        infiniopStatus_t calculate(                       \
            void *workspace,                              \
            size_t workspace_size,                        \
            void *c,                                      \
            float beta,                                   \
            void const *a,                                \
            void const *b,                                \
            float alpha,                                  \
            void *stream) const;                          \
    };                                                    \
    }

#endif // __MATMUL_H__
