#ifndef __MATMUL_H__
#define __MATMUL_H__

#include "blas.h"
#include "infiniop/operator.h"

#define DESCRIPTOR(NAMESPACE, HANDLE)                     \
                                                          \
    namespace matmul::NAMESPACE {                         \
    class Descriptor final : public InfiniopDescriptor {  \
        struct Opaque;                                    \
        Opaque *_opaque;                                  \
                                                          \
        Descriptor(                                       \
            infiniDtype_t dtype_,                         \
            MatmulInfo info_,                             \
            size_t workspace_size_,                       \
            Opaque *opaque,                               \
            infiniDevice_t device_type,                   \
            int device_id)                                \
            : InfiniopDescriptor{device_type, device_id}, \
              _opaque(opaque),                            \
              dtype(dtype_),                              \
              info(info_),                                \
              workspace_size(workspace_size_) {}          \
                                                          \
    public:                                               \
        infiniDtype_t dtype;                              \
        MatmulInfo info;                                  \
        size_t workspace_size;                            \
                                                          \
        ~Descriptor();                                    \
                                                          \
        static infiniopStatus_t create(                   \
            HANDLE handle,                                \
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
