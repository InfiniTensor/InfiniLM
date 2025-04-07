#ifndef __INFINIOP_ELEMENTWISE_H__
#define __INFINIOP_ELEMENTWISE_H__

#include "../operator.h"
#include "../tensor.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#define ELEMENTWISE_DESCRIPTOR(OP, NAMESPACE)                                 \
                                                                              \
    namespace op::OP::NAMESPACE {                                             \
    class Descriptor final : public InfiniopDescriptor {                      \
        infiniDtype_t _dtype;                                                 \
        op::elementwise::ElementwiseInfo _info;                               \
        std::unique_ptr<op::elementwise::NAMESPACE::DeviceImpl> _device_info; \
                                                                              \
        Descriptor(                                                           \
            infiniDtype_t dtype,                                              \
            op::elementwise::ElementwiseInfo info,                            \
            op::elementwise::NAMESPACE::DeviceImpl *device_info,              \
            infiniDevice_t device_type,                                       \
            int device_id)                                                    \
            : InfiniopDescriptor{device_type, device_id},                     \
              _dtype(dtype),                                                  \
              _info(std::move(info)),                                         \
              _device_info(device_info) {}                                    \
                                                                              \
    public:                                                                   \
        ~Descriptor();                                                        \
                                                                              \
        static infiniStatus_t create(                                         \
            infiniopHandle_t handle,                                          \
            Descriptor **desc_ptr,                                            \
            infiniopTensorDescriptor_t output_desc,                           \
            std::vector<infiniopTensorDescriptor_t> input_descs);             \
                                                                              \
        infiniStatus_t calculate(                                             \
            void *output,                                                     \
            std::vector<const void *> inputs,                                 \
            void *stream) const;                                              \
    };                                                                        \
    }

namespace op::elementwise {

// struct that stores data needed for elementwise operation
struct ElementwiseInfo {
    size_t output_size;
    size_t ndim;
    bool output_contiguous;
    bool *input_contiguous;
    bool *input_broadcasted;
    size_t *output_shape;
    size_t **input_shapes;
    ptrdiff_t *output_strides;
    ptrdiff_t **input_strides;
    size_t input_size;

    ElementwiseInfo() = default;

    // Destructor to free allocated memory
    ~ElementwiseInfo() {
        delete[] input_contiguous;
        delete[] input_broadcasted;
        delete[] output_shape;
        delete[] output_strides;

        for (size_t i = 0; i < input_size; ++i) {
            delete[] input_shapes[i];
            delete[] input_strides[i];
        }
        delete[] input_shapes;
        delete[] input_strides;
    }

    ElementwiseInfo(const ElementwiseInfo &other)
        : output_size(other.output_size),
          ndim(other.ndim),
          output_contiguous(other.output_contiguous),
          input_size(other.input_size) {

        input_contiguous = new bool[input_size];
        std::memcpy(input_contiguous, other.input_contiguous, input_size * sizeof(*input_contiguous));

        input_broadcasted = new bool[input_size];
        std::memcpy(input_broadcasted, other.input_broadcasted, input_size * sizeof(*input_broadcasted));

        output_shape = new size_t[ndim];
        std::memcpy(output_shape, other.output_shape, ndim * sizeof(*output_shape));

        output_strides = new ptrdiff_t[ndim];
        std::memcpy(output_strides, other.output_strides, ndim * sizeof(*output_strides));

        input_shapes = new size_t *[input_size];
        for (size_t i = 0; i < input_size; ++i) {
            input_shapes[i] = new size_t[ndim];
            std::memcpy(input_shapes[i], other.input_shapes[i], ndim * sizeof(*input_shapes[i]));
        }

        input_strides = new ptrdiff_t *[input_size];
        for (size_t i = 0; i < input_size; ++i) {
            input_strides[i] = new ptrdiff_t[ndim];
            std::memcpy(input_strides[i], other.input_strides[i], ndim * sizeof(*input_strides[i]));
        }
    }

    ElementwiseInfo(ElementwiseInfo &&other) noexcept
        : output_size(other.output_size),
          ndim(other.ndim),
          output_contiguous(other.output_contiguous),
          input_contiguous(other.input_contiguous),
          input_broadcasted(other.input_broadcasted),
          output_shape(other.output_shape),
          input_shapes(other.input_shapes),
          output_strides(other.output_strides),
          input_strides(other.input_strides),
          input_size(other.input_size) {
        other.input_contiguous = nullptr;
        other.input_broadcasted = nullptr;
        other.output_shape = nullptr;
        other.input_shapes = nullptr;
        other.output_strides = nullptr;
        other.input_strides = nullptr;
        other.input_size = 0;
    }

    ElementwiseInfo &operator=(const ElementwiseInfo &other) = delete;
    ElementwiseInfo &operator=(ElementwiseInfo &&other) = delete;
};

inline infiniStatus_t createElementwiseInfo(
    ElementwiseInfo &info,
    infiniopTensorDescriptor_t output_desc,
    std::vector<infiniopTensorDescriptor_t> input_descs) {

    if (!output_desc || input_descs.empty()) {
        return INFINI_STATUS_BAD_PARAM;
    }

    // Destination cannot have broadcast setup
    if (output_desc->hasBroadcastDim()) {
        return INFINI_STATUS_BAD_TENSOR_STRIDES;
    }

    info.input_size = input_descs.size();
    info.ndim = output_desc->ndim();
    info.output_size = output_desc->numel();
    info.output_contiguous = output_desc->isContiguous();

    // Allocate memory for arrays
    info.input_contiguous = new bool[info.input_size];
    info.input_broadcasted = new bool[info.input_size];
    info.output_shape = new size_t[info.ndim];
    info.output_strides = new ptrdiff_t[info.ndim];
    info.input_shapes = new size_t *[info.input_size];
    info.input_strides = new ptrdiff_t *[info.input_size];

    // Fill arrays
    const auto output_shape = output_desc->shape();
    const auto output_strides = output_desc->strides();
    std::memcpy(info.output_shape, output_shape.data(), info.ndim * sizeof(*info.output_shape));
    std::memcpy(info.output_strides, output_strides.data(), info.ndim * sizeof(*info.output_strides));

    for (size_t i = 0; i < info.input_size; ++i) {
        auto &desc = input_descs[i];
        info.input_contiguous[i] = desc->isContiguous();
        info.input_broadcasted[i] = !info.input_contiguous[i] && (desc->ndim() != info.ndim || desc->hasBroadcastDim());

        info.input_shapes[i] = new size_t[desc->ndim()];
        const auto &in_shape = desc->shape();
        std::memcpy(info.input_shapes[i], in_shape.data(), desc->ndim() * sizeof(*info.input_shapes[i]));

        info.input_strides[i] = new ptrdiff_t[desc->ndim()];
        const auto &in_strides = desc->strides();
        std::memcpy(info.input_strides[i], in_strides.data(), desc->ndim() * sizeof(*info.input_strides[i]));
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::elementwise

#endif // __INFINIOP_ELEMENTWISE_H__
