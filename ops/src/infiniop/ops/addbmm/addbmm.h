#ifndef ADDBMM_H
#define ADDBMM_H

#include "../../operator.h"
#include "info.h" // 对应 addbmm_info.h

// 宏定义：用于生成不同命名空间下的 Descriptor 类
// 注意：addbmm 需要处理 alpha, beta 和多个输入张量
#define DESCRIPTOR(NAMESPACE)                                           \
    namespace op::addbmm::NAMESPACE {                                   \
    class Descriptor final : public InfiniopDescriptor {                \
        struct Opaque;                                                  \
        Opaque *_opaque;                                                \
        AddbmmInfo _info;                                               \
        size_t _workspace_size;                                         \
                                                                        \
        Descriptor(                                                     \
            Opaque *opaque,                                             \
            AddbmmInfo info,                                            \
            size_t workspace_size,                                      \
            infiniDevice_t device_type,                                 \
            int device_id)                                              \
            : InfiniopDescriptor{device_type, device_id},               \
              _opaque(opaque),                                          \
              _info(info),                                              \
              _workspace_size(workspace_size) {}                        \
                                                                        \
    public:                                                             \
        ~Descriptor();                                                  \
                                                                        \
        size_t workspaceSize() const { return _workspace_size; }        \
                                                                        \
        /* create 函数接收 Tensor 描述符列表和标量系数 */ \
        static infiniStatus_t create(                                   \
            infiniopHandle_t handle,                                    \
            Descriptor **desc_ptr,                                      \
            infiniopTensorDescriptor_t out_desc,                        \
            std::vector<infiniopTensorDescriptor_t> input_desc_vec,     \
            float alpha,                                                \
            float beta);                                                \
                                                                        \
        /* calculate 函数接收数据指针列表 */                  \
        infiniStatus_t calculate(                                       \
            void *workspace,                                            \
            size_t workspace_size,                                      \
            void *output,                                               \
            std::vector<const void *> inputs,                           \
            void *stream) const;                                        \
    };                                                                  \
    }

#endif // ADDBMM_H
