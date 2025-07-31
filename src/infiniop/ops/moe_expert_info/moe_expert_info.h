#ifndef __MOE_EXPERT_INFO_H__
#define __MOE_EXPERT_INFO_H__

#include "../../operator.h"
#include "info.h"

/**
 * @brief Defines the structure for a backend-specific descriptor class.
 *
 * This macro generates the necessary class definition for a given backend
 * (NAMESPACE), encapsulating the implementation details for the
 * MoEExpertInfo operator.
 */
#define DESCRIPTOR(NAMESPACE)                                                      \
                                                                                   \
    namespace op::moe_expert_info::NAMESPACE {                                     \
    class Descriptor final : public InfiniopDescriptor {                           \
        struct Opaque;                                                             \
        Opaque *_opaque;                                                           \
        MoEExpertInfoInfo _info;                                                   \
                                                                                   \
        /* Private constructor, called by the static create method */              \
        Descriptor(                                                                \
            MoEExpertInfoInfo info,                                                \
            Opaque *opaque,                                                        \
            infiniDevice_t device_type,                                            \
            int device_id)                                                         \
            : InfiniopDescriptor{device_type, device_id},                          \
              _opaque(opaque),                                                     \
              _info(info) {}                                                       \
                                                                                   \
      public:                                                                      \
        /* Destructor to clean up backend-specific resources */                    \
        ~Descriptor();                                                             \
                                                                                   \
        /** \
         * @brief Creates a backend-specific descriptor for the operator.          \
         */                                                                        \
        static infiniStatus_t create(                                              \
            infiniopHandle_t handle,                                               \
            Descriptor **desc_ptr,                                                 \
            infiniopTensorDescriptor_t topk_ind_desc,                              \
            infiniopTensorDescriptor_t expert_counts_desc,                         \
            infiniopTensorDescriptor_t expert_offsets_desc);                       \
                                                                                   \
        /** \
         * @brief Executes the calculation on the given stream.                    \
         */                                                                        \
        infiniStatus_t calculate(                                                  \
            const void *topk_ind,                                                  \
            void *expert_counts,                                                   \
            void *expert_offsets,                                                  \
            void *stream) const;                                                   \
    };                                                                             \
    }

#endif // __MOE_EXPERT_INFO_H__