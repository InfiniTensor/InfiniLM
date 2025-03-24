#ifndef __INFINIOP_BINARY_CPU_H__
#define __INFINIOP_BINARY_CPU_H__

#include "../../devices/cpu/common_cpu.h"
#include "../../operator.h"
#include "../../tensor.h"
#include <array>
#include <numeric>
#include <utility>

/**
 * 该类的设计基于 matmul.h 中 YdrMaster 设计的 DESCRIPTOR 宏。
 */

#define BINARY_DESCRIPTOR(OP, NAMESPACE)                   \
                                                           \
    namespace op::OP::NAMESPACE {                          \
    class Descriptor final : public InfiniopDescriptor {   \
        struct Opaque;                                     \
        Opaque *_opaque;                                   \
        infiniDtype_t _dtype;                              \
        op::common_cpu::binary_op::BinaryCpuInfo _info;    \
                                                           \
        Descriptor(                                        \
            infiniDtype_t dtype,                           \
            op::common_cpu::binary_op::BinaryCpuInfo info, \
            Opaque *opaque,                                \
            infiniDevice_t device_type,                    \
            int device_id)                                 \
            : InfiniopDescriptor{device_type, device_id},  \
              _opaque(opaque),                             \
              _dtype(dtype),                               \
              _info(info) {}                               \
                                                           \
    public:                                                \
        ~Descriptor();                                     \
                                                           \
        static infiniStatus_t create(                      \
            infiniopHandle_t handle,                       \
            Descriptor **desc_ptr,                         \
            infiniopTensorDescriptor_t c_desc,             \
            infiniopTensorDescriptor_t a_desc,             \
            infiniopTensorDescriptor_t b_desc);            \
                                                           \
        infiniStatus_t calculate(                          \
            void *c,                                       \
            const void *a,                                 \
            const void *b,                                 \
            void *stream) const;                           \
    };                                                     \
    }

namespace op::common_cpu {

namespace binary_op {

// Stores metadata for binary operations on CPU
struct BinaryCpuInfo {
    size_t c_data_size;
    size_t ndim;
    bool broadcasted;
    std::vector<size_t> c_shape;
    std::vector<size_t> a_shape;
    std::vector<size_t> b_shape;
    std::vector<ptrdiff_t> c_strides;
    std::vector<ptrdiff_t> a_strides;
    std::vector<ptrdiff_t> b_strides;

    BinaryCpuInfo(size_t c_data_size,
                  size_t ndim,
                  bool broadcasted,
                  std::vector<size_t> c_shape,
                  std::vector<size_t> a_shape,
                  std::vector<size_t> b_shape,
                  std::vector<ptrdiff_t> c_strides,
                  std::vector<ptrdiff_t> a_strides,
                  std::vector<ptrdiff_t> b_strides)
        : c_data_size(c_data_size),
          ndim(ndim),
          broadcasted(broadcasted),
          c_shape(std::move(c_shape)),
          a_shape(std::move(a_shape)),
          b_shape(std::move(b_shape)),
          c_strides(std::move(c_strides)),
          a_strides(std::move(a_strides)),
          b_strides(std::move(b_strides)) {}

    BinaryCpuInfo(infiniopTensorDescriptor_t c_desc,
                  infiniopTensorDescriptor_t a_desc,
                  infiniopTensorDescriptor_t b_desc)
        : ndim(c_desc->ndim()),
          c_shape(std::move(c_desc->shape())),
          a_shape(std::move(a_desc->shape())),
          b_shape(std::move(b_desc->shape())),
          c_strides(std::move(c_desc->strides())),
          a_strides(std::move(a_desc->strides())),
          b_strides(std::move(b_desc->strides())) {
        this->c_data_size = std::accumulate(c_shape.begin(), c_shape.end(), 1ULL, std::multiplies<size_t>());
        this->broadcasted = (a_strides != c_strides) || (b_strides != c_strides);
    }
};

// Helper function for compile-time optimized checks
template <size_t N>
bool isDtypeSupported(infiniDtype_t dtype, const std::array<infiniDtype_t, N> &supported_dtypes) {
    for (size_t i = 0; i < N; ++i) {
        if (dtype == supported_dtypes[i]) {
            return true;
        }
    }
    return false;
}

// Checks if the tensors are compatible for binary operations based on dtype and shape requirements.
template <size_t N>
infiniStatus_t check(infiniopTensorDescriptor_t c_desc,
                     infiniopTensorDescriptor_t a_desc,
                     infiniopTensorDescriptor_t b_desc,
                     const std::array<infiniDtype_t, N> &supported_dtypes,
                     bool require_same_dtype,
                     bool require_same_shape) {
    const auto dtype = c_desc->dtype();

    if (!isDtypeSupported(dtype, supported_dtypes)) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    // check dtype match if required
    if (require_same_dtype && (a_desc->dtype() != dtype || b_desc->dtype() != dtype)) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    // check shape compatibility if required
    if (require_same_shape) {
        const auto ndim = c_desc->ndim();
        if (a_desc->ndim() != ndim || b_desc->ndim() != ndim) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        const auto &c_shape = c_desc->shape();
        const auto &a_shape = a_desc->shape();
        const auto &b_shape = b_desc->shape();
        for (size_t i = 0; i < ndim; ++i) {
            if (c_shape[i] != a_shape[i] || c_shape[i] != b_shape[i]) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        }
    }

    return INFINI_STATUS_SUCCESS;
}

// Perform binary computation
template <typename Tdata, typename BinaryOp, typename... Args>
void calculate(BinaryCpuInfo info, void *c, const void *a, const void *b, Args &&...args) {
    auto a_ = reinterpret_cast<const Tdata *>(a);
    auto b_ = reinterpret_cast<const Tdata *>(b);
    auto c_ = reinterpret_cast<Tdata *>(c);
    ssize_t data_size = info.c_data_size;

#pragma omp parallel for
    for (ssize_t i = 0; i < data_size; ++i) {
        size_t a_index = info.broadcasted ? indexToReducedOffset(i, info.ndim, info.c_strides.data(), info.a_strides.data())
                                          : indexToOffset(i, info.ndim, info.a_shape.data(), info.a_strides.data());
        size_t b_index = info.broadcasted ? indexToReducedOffset(i, info.ndim, info.c_strides.data(), info.b_strides.data())
                                          : indexToOffset(i, info.ndim, info.b_shape.data(), info.b_strides.data());
        size_t c_index = indexToOffset(i, info.ndim, info.c_shape.data(), info.c_strides.data());

        if constexpr (std::is_same_v<Tdata, fp16_t>) {
            float a_val = utils::cast<float>(a_[a_index]);
            float b_val = utils::cast<float>(b_[b_index]);
            c_[c_index] = utils::cast<fp16_t>(BinaryOp{}(a_val, b_val, std::forward<Args>(args)...));
        } else {
            c_[c_index] = BinaryOp{}(a_[a_index], b_[b_index], std::forward<Args>(args)...);
        }
    }
}

} // namespace binary_op
} // namespace op::common_cpu

#endif // __INFINIOP_BINARY_CPU_H__
