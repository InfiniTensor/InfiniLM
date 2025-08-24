#ifndef __GEMM_H__
#define __GEMM_H__

#include "../../operator.h"
#include "info.h"

/**
 * # 关于 `DESCRIPTOR(NAMESPACE)` 和 `struct Opaque;` 的说明
 *
 * > - Data: 2025/02/20
 * > - Author: <YdrMaster ydrml@hotmail.com>
 *
 * `DESCRIPTOR(NAMESPACE)` 宏定义了一个 `Descriptor` 类，
 * 该类继承自 `InfiniopDescriptor`，并在内部定义了：
 *
 * - 描述任何硬件上的矩阵乘算子必要的信息
 *   - 参数数据类型；
 *   - 矩阵乘运算信息（BMNK）；
 *   - 工作空间大小；
 * - 特定硬件独有的描述信息保存在 `struct Opaque;` 中；
 * - 私有的构造函数；
 * - 公共接口
 *   - 析构函数；
 *   - 静态的工厂函数；
 *   - 矩阵乘计算函数；
 *
 * 这个宏必须写成一个宏，因为静态成员函数是不可继承的，但每个不同硬件上有不同的实现。
 * 使用宏声明的不同硬件的矩阵乘描述符是不相关类型（并且它们也不必相关）。
 *
 * 这些描述符的声明对 operator.cc 可见，而 operator.cc 文件编译到 infiniop 库的硬件无关部分中，
 * 因此，描述符声明不可以出现硬件相关的类型。
 *
 * 为了隐藏必须伴随算子描述符保存的硬件相关类型（例如，Ascend 上的 `executor`），
 * 这里使用了 `Opaque` 声明。
 *
 * Opaque 表示“不透明的”，即其内部结构面型头文件的使用者隐藏。
 * 这种设计来自 C++ 的一种设计模式，其经典形式称为 [PImpl](https://zh.cppreference.com/w/cpp/language/pimpl) 模式。
 * 由于此处仅需要结构体的指针，因此结构体可以仅声明不定义，从而隐藏了结构体的成员。
 * 通过将结构体定义为 private，保证了外部不可能操作这个指针，
 * 因此这个设计可以由类型自由控制其成员的生命周期，
 * 同时不必将成员的形式暴露给头文件的使用者，
 * 这是一种安全的封装。
 *
 * 这个宏仅适用于矩阵乘，但这种模式很容易复制到其他算子，以简化和规范算子的声明。
 */

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::gemm::NAMESPACE {                              \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        infiniDtype_t _dtype;                                    \
        MatmulInfo _info;                                        \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            infiniDtype_t dtype,                                 \
            MatmulInfo info,                                     \
            size_t workspace_size_,                              \
            Opaque *opaque,                                      \
            infiniDevice_t device_type,                          \
            int device_id)                                       \
            : InfiniopDescriptor{device_type, device_id},        \
              _opaque(opaque),                                   \
              _dtype(dtype),                                     \
              _info(info),                                       \
              _workspace_size(workspace_size_) {}                \
                                                                 \
    public:                                                      \
        ~Descriptor();                                           \
                                                                 \
        size_t workspaceSize() const { return _workspace_size; } \
                                                                 \
        static infiniStatus_t create(                            \
            infiniopHandle_t handle,                             \
            Descriptor **desc_ptr,                               \
            infiniopTensorDescriptor_t c_desc,                   \
            infiniopTensorDescriptor_t a_desc,                   \
            infiniopTensorDescriptor_t b_desc);                  \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace, size_t workspace_size,              \
            void *c,                                             \
            float beta,                                          \
            const void *a,                                       \
            const void *b,                                       \
            float alpha,                                         \
            void *stream) const;                                 \
    };                                                           \
    }

#endif // __GEMM_H__
