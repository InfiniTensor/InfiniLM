#ifndef __MOE_COMBINE_H__
#define __MOE_COMBINE_H__

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                                  \
                                                                               \
    namespace op::moe_combine::NAMESPACE {                                     \
    class Descriptor final : public InfiniopDescriptor {                       \
        struct Opaque;                                                         \
        Opaque *_opaque;                                                       \
		infiniDtype_t _dtype;                                                  \
        MoECombineInfo _info;                                                  \
        size_t _workspace_size;                                                \
                                                                               \
        Descriptor(                                                            \
			infiniDtype_t dtype,                                               \
            MoECombineInfo info,                                               \
			size_t workspace_size,                                             \
            Opaque *opaque,                                                    \
            infiniDevice_t device_type,                                        \
            int device_id)                                                     \
            : InfiniopDescriptor{device_type, device_id},                      \
				_opaque(opaque),                                               \
				_dtype(dtype),                                                 \
				_info(info),                                                   \
				_workspace_size(workspace_size) {}                             \
                                                                               \
    public:                                                                    \
        ~Descriptor();                                                         \
                                                                               \
        static infiniStatus_t create(                                          \
            infiniopHandle_t handle,                                           \
            Descriptor **desc_ptr,                                             \
            infiniopTensorDescriptor_t permuted_input_desc,                    \
            infiniopTensorDescriptor_t gating_weights_desc,                    \
            infiniopTensorDescriptor_t aux_info_desc,                          \
            infiniopTensorDescriptor_t output_desc);                           \
                                                                               \
        infiniStatus_t calculate(                                              \
            const void *permuted_input,                                        \
            const void *gating_weights,                                        \
            const void *aux_info,                                              \
            void *output,                                                      \
            void *stream) const;                                               \
    };                                                                         \
    }

#endif // __MOE_COMBINE_H__ 