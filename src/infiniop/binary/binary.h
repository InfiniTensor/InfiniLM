#ifndef __INFINIOP_BINARY_H__
#define __INFINIOP_BINARY_H__

#include "../devices/cpu/common_cpu.h"
#include "../operator.h"
#include "../tensor.h"
#include <numeric>

/**
 * 该类的设计基于 matmul.h 中 YdrMaster 设计的 DESCRIPTOR 宏。
 */

#define BINARY_DESCRIPTOR(OP, NAMESPACE)                  \
                                                          \
    namespace op::OP::NAMESPACE {                         \
    class Descriptor final : public InfiniopDescriptor {  \
        struct Opaque;                                    \
        Opaque *_opaque;                                  \
        infiniDtype_t _dtype;                             \
        op::binary::BinaryInfo _info;                     \
                                                          \
        Descriptor(                                       \
            infiniDtype_t dtype,                          \
            op::binary::BinaryInfo info,                  \
            Opaque *opaque,                               \
            infiniDevice_t device_type,                   \
            int device_id)                                \
            : InfiniopDescriptor{device_type, device_id}, \
              _opaque(opaque),                            \
              _dtype(dtype),                              \
              _info(info) {}                              \
                                                          \
    public:                                               \
        ~Descriptor();                                    \
                                                          \
        static infiniStatus_t create(                     \
            infiniopHandle_t handle,                      \
            Descriptor **desc_ptr,                        \
            infiniopTensorDescriptor_t c_desc,            \
            infiniopTensorDescriptor_t a_desc,            \
            infiniopTensorDescriptor_t b_desc);           \
                                                          \
        infiniStatus_t calculate(                         \
            void *c,                                      \
            const void *a,                                \
            const void *b,                                \
            void *stream) const;                          \
    };                                                    \
    }


namespace op::binary {

// Stores metadata for binary operations on CPU
struct BinaryInfo {
private:
    BinaryInfo(infiniopTensorDescriptor_t c_desc,
               infiniopTensorDescriptor_t a_desc,
               infiniopTensorDescriptor_t b_desc)
        : ndim(c_desc->ndim()),
          c_shape(std::move(c_desc->shape())),
          a_shape(std::move(a_desc->shape())),
          b_shape(std::move(b_desc->shape())),
          c_strides(std::move(c_desc->strides())),
          a_strides(std::move(a_desc->strides())),
          b_strides(std::move(b_desc->strides())) {
        this->c_data_size = std::accumulate(c_shape.begin(), c_shape.end(), size_t(1), std::multiplies<size_t>());
        this->broadcasted = (a_strides != c_strides) || (b_strides != c_strides);
    }

public:
    size_t c_data_size;
    size_t ndim;
    bool broadcasted;
    std::vector<size_t> c_shape;
    std::vector<size_t> a_shape;
    std::vector<size_t> b_shape;
    std::vector<ptrdiff_t> c_strides;
    std::vector<ptrdiff_t> a_strides;
    std::vector<ptrdiff_t> b_strides;

    static infiniStatus_t create(
        BinaryInfo **instance,
        infiniopTensorDescriptor_t c_desc,
        infiniopTensorDescriptor_t a_desc,
        infiniopTensorDescriptor_t b_desc) {
        if (!c_desc || !a_desc || !b_desc) {
            return INFINI_STATUS_BAD_PARAM;
        }

        *instance = new BinaryInfo(c_desc, a_desc, b_desc);
        return INFINI_STATUS_SUCCESS;
    }
};
} // namespace op::binary

#endif // __INFINIOP_BINARY_H__
