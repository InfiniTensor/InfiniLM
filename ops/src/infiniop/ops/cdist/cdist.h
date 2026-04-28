#ifndef __CDIST_H__
#define __CDIST_H__

#include "../../operator.h"
#include "info.h"

/**
 * # 关于 `cdist` 算子描述符的说明
 * * 仿照 GEMM 的 PImpl (Opaque) 设计模式，将硬件相关的执行上下文（如 CUDA Handle、计算流等）
 * 隐藏在 `Opaque` 结构体中，确保头文件在不同后端（CPU/NVIDIA/Ascend）间的一致性。
 */

#define DESCRIPTOR(NAMESPACE)                                       \
                                                                    \
    namespace op::cdist::NAMESPACE {                                \
    class Descriptor final : public InfiniopDescriptor {            \
        struct Opaque;                                              \
        Opaque *_opaque;                                            \
        infiniDtype_t _dtype;                                       \
        CdistInfo _info; /* 包含 M, N, D 维度信息 */          \
        size_t _workspace_size;                                     \
        double _p; /* 范数阶数，创建时固定 */             \
                                                                    \
        Descriptor(                                                 \
            infiniDtype_t dtype,                                    \
            CdistInfo info,                                         \
            double p,                                               \
            size_t workspace_size_,                                 \
            Opaque *opaque,                                         \
            infiniDevice_t device_type,                             \
            int device_id)                                          \
            : InfiniopDescriptor{device_type, device_id},           \
              _opaque(opaque),                                      \
              _dtype(dtype),                                        \
              _info(info),                                          \
              _workspace_size(workspace_size_),                     \
              _p(p) {}                                              \
                                                                    \
    public:                                                         \
        ~Descriptor();                                              \
                                                                    \
        size_t workspaceSize() const { return _workspace_size; }    \
                                                                    \
        static infiniStatus_t create(                               \
            infiniopHandle_t handle,                                \
            Descriptor **desc_ptr,                                  \
            infiniopTensorDescriptor_t y_desc,  /* 输出 (M, N) */ \
            infiniopTensorDescriptor_t x1_desc, /* 输入 (M, D) */ \
            infiniopTensorDescriptor_t x2_desc, /* 输入 (N, D) */ \
            double p);                                              \
                                                                    \
        infiniStatus_t calculate(                                   \
            void *workspace,                                        \
            size_t workspace_size,                                  \
            void *y, /* 结果矩阵 */                             \
            const void *x1,                                         \
            const void *x2,                                         \
            void *stream) const;                                    \
    };                                                              \
    }

#endif // __CDIST_H__
