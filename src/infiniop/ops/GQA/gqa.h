#ifndef __GQA_H__
#define __GQA_H__

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                                  \
                                                                               \
    namespace op::gqa::NAMESPACE {                                             \
    class Descriptor final : public InfiniopDescriptor {                       \
        struct Opaque;                                                         \
        Opaque *_opaque;                                                       \
		infiniDtype_t _dtype;                                                  \
        GQAInfo _info;                                                         \
        size_t _workspace_size;                                                \
                                                                               \
        Descriptor(                                                            \
			infiniDtype_t dtype,                                               \
            GQAInfo info,                                               	   \
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
            infiniopTensorDescriptor_t q_desc,                                 \
			infiniopTensorDescriptor_t k_desc,                                 \
			infiniopTensorDescriptor_t v_desc,                                 \
			infiniopTensorDescriptor_t output_desc);                           \
                                                                               \
        infiniStatus_t calculate(                                              \
            const void *q, 													   \
			const void *k, 													   \
			const void *v,                                     				   \
			void *output,                                                      \
			void *stream) const;                                               \
    };                                                                         \
    }

#endif 