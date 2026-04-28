#ifndef __BINARY_CROSS_ENTROPY_WITH_LOGITS_H__
#define __BINARY_CROSS_ENTROPY_WITH_LOGITS_H__

#include "../../operator.h"
#include "info.h"

/**
 * # 关于 `BCEWithLogits` 算子描述符的说明
 * * 采用 PImpl 设计模式，将不同硬件后端（如 CUDA 原生算子、CPU 循环、或是芯片厂商的专用库调用）
 * 封装在 `Opaque` 结构中。
 * * 描述符在创建时会完成形状校验、步长分析，并确定最优的计算 Workspace 大小。
 */

#define DESCRIPTOR(NAMESPACE)                                                        \
                                                                                     \
    namespace op::bce_with_logits::NAMESPACE {                                       \
    class Descriptor final : public InfiniopDescriptor {                             \
        struct Opaque;                                                               \
        Opaque *_opaque;                                                             \
        infiniDtype_t _dtype;                                                        \
        BCEWithLogitsInfo _info; /* 包含各输入输出张量的维度与步长 */ \
        size_t _workspace_size;                                                      \
        infiniopReduction_t _reduction;                                              \
                                                                                     \
        Descriptor(                                                                  \
            infiniDtype_t dtype,                                                     \
            BCEWithLogitsInfo info,                                                  \
            infiniopReduction_t reduction,                                           \
            size_t workspace_size_,                                                  \
            Opaque *opaque,                                                          \
            infiniDevice_t device_type,                                              \
            int device_id)                                                           \
            : InfiniopDescriptor{device_type, device_id},                            \
              _opaque(opaque),                                                       \
              _dtype(dtype),                                                         \
              _info(info),                                                           \
              _workspace_size(workspace_size_),                                      \
              _reduction(reduction) {}                                               \
                                                                                     \
    public:                                                                          \
        ~Descriptor();                                                               \
                                                                                     \
        size_t workspaceSize() const { return _workspace_size; }                     \
                                                                                     \
        static infiniStatus_t create(                                                \
            infiniopHandle_t handle,                                                 \
            Descriptor **desc_ptr,                                                   \
            infiniopTensorDescriptor_t out_desc,                                     \
            infiniopTensorDescriptor_t logits_desc,                                  \
            infiniopTensorDescriptor_t target_desc,                                  \
            infiniopTensorDescriptor_t weight_desc,                                  \
            infiniopTensorDescriptor_t pos_weight_desc,                              \
            infiniopReduction_t reduction);                                          \
                                                                                     \
        infiniStatus_t calculate(                                                    \
            void *workspace,                                                         \
            size_t workspace_size,                                                   \
            void *out,                                                               \
            const void *logits,                                                      \
            const void *target,                                                      \
            const void *weight,     /* 可选，可为 nullptr */                    \
            const void *pos_weight, /* 可选，可为 nullptr */                    \
            void *stream) const;                                                     \
    };                                                                               \
    }

#endif // __BINARY_CROSS_ENTROPY_WITH_LOGITS_H__
