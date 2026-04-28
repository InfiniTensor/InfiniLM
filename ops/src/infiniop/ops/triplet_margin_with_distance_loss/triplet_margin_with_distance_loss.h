#ifndef __TRIPLET_MARGIN_WITH_DISTANCE_LOSS_H__
#define __TRIPLET_MARGIN_WITH_DISTANCE_LOSS_H__

#include "../../operator.h"
#include "info.h"
#define DESCRIPTOR(NAMESPACE)                                    \
    namespace op::triplet_margin_with_distance_loss::NAMESPACE { \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        TripletMarginWithDistanceLossInfo _info;                 \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            Opaque *opaque,                                      \
            TripletMarginWithDistanceLossInfo info,              \
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
                                                                 \
        size_t workspaceSize() const { return _workspace_size; } \
                                                                 \
        static infiniStatus_t create(                            \
            infiniopHandle_t handle,                             \
            Descriptor **desc_ptr,                               \
            infiniopTensorDescriptor_t output_desc,              \
            infiniopTensorDescriptor_t anchor_desc,              \
            infiniopTensorDescriptor_t positive_desc,            \
            infiniopTensorDescriptor_t negative_desc,            \
            float margin,                                        \
            int swap,                                            \
            int reduction);                                      \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace,                                     \
            size_t workspace_size,                               \
            void *output,                                        \
            const void *anchor,                                  \
            const void *positive,                                \
            const void *negative,                                \
            void *stream) const;                                 \
    };                                                           \
    }

#endif // __TRIPLET_MARGIN_WITH_DISTANCE_LOSS_H__
