#ifndef __MOE_DISPATCH_H__
#define __MOE_DISPATCH_H__

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                                  \\
                                                                               \\
    namespace op::moe_dispatch::NAMESPACE {                                    \\
    class Descriptor final : public InfiniopDescriptor {                       \\
        struct Opaque;                                                         \\
        Opaque *_opaque;                                                       \\
        MoEDispatchInfo _info;                                                 \\
                                                                               \\
        Descriptor(                                                            \\
            MoEDispatchInfo info,                                              \\
            Opaque *opaque,                                                    \\
            infiniDevice_t device_type,                                        \\
            int device_id)                                                     \\
            : InfiniopDescriptor{device_type, device_id},                      \\
              _opaque(opaque),                                                 \\
              _info(info) {}                                                   \\
                                                                               \\
    public:                                                                    \\
        ~Descriptor();                                                         \\
                                                                               \\
        static infiniStatus_t create(                                          \\
            infiniopHandle_t handle,                                           \\
            Descriptor **desc_ptr,                                             \\
            int num_experts,                                                   \\
            infiniopTensorDescriptor_t input_desc,                             \\
            infiniopTensorDescriptor_t indices_desc,                           \\
            infiniopTensorDescriptor_t permuted_output_desc,                   \\
            infiniopTensorDescriptor_t aux_info_desc);                         \\
                                                                               \\
        infiniStatus_t calculate(                                              \\
            const void *input,                                                 \\
            const void *indices,                                               \\
            void *permuted_output,                                             \\
            void *aux_info,                                                    \\
            void *stream) const;                                               \\
    };                                                                         \\
    }

#endif // __MOE_DISPATCH_H__ 