#ifndef __INFINIOP_BINARY_H__
#define __INFINIOP_BINARY_H__

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
    size_t c_data_size;
    size_t ndim;
    bool contiguous;
    bool broadcasted;
    std::vector<size_t> c_shape;
    std::vector<size_t> a_shape;
    std::vector<size_t> b_shape;
    std::vector<ptrdiff_t> c_strides;
    std::vector<ptrdiff_t> a_strides;
    std::vector<ptrdiff_t> b_strides;
};

inline infiniStatus_t createBinaryInfo(BinaryInfo &info,
                                       infiniopTensorDescriptor_t c_desc,
                                       infiniopTensorDescriptor_t a_desc,
                                       infiniopTensorDescriptor_t b_desc) {

    if (!c_desc || !a_desc || !b_desc) {
        return INFINI_STATUS_BAD_PARAM;
    }

    info.c_data_size = c_desc->numel();
    info.ndim = c_desc->ndim();
    info.contiguous = c_desc->isContiguous() && a_desc->isContiguous() && b_desc->isContiguous();

    // Destination cannot have broadcast setup
    if (c_desc->hasBroadcastDim()) {
        return INFINI_STATUS_BAD_TENSOR_STRIDES;
    }
    const bool ndim_match = (c_desc->ndim() == a_desc->ndim()) && (c_desc->ndim() == b_desc->ndim());
    info.broadcasted = !info.contiguous && (!ndim_match || a_desc->hasBroadcastDim() || b_desc->hasBroadcastDim());

    info.c_shape = std::move(c_desc->shape());
    info.a_shape = std::move(a_desc->shape());
    info.b_shape = std::move(b_desc->shape());
    info.c_strides = std::move(c_desc->strides());
    info.a_strides = std::move(a_desc->strides());
    info.b_strides = std::move(b_desc->strides());

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::binary

#endif // __INFINIOP_BINARY_H__
