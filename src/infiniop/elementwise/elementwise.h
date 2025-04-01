#ifndef __INFINIOP_ELEMENTWISE_H__
#define __INFINIOP_ELEMENTWISE_H__

#include "../operator.h"
#include "../tensor.h"
#include <algorithm>
#include <memory>
#include <numeric>
#include <vector>

#define DEVICE_IMPL(NAMESPACE)                                                   \
                                                                                 \
    namespace op::elementwise::NAMESPACE {                                       \
    class DeviceImpl final {                                                     \
        struct Opaque;                                                           \
        std::unique_ptr<Opaque> _opaque;                                         \
                                                                                 \
        DeviceImpl(Opaque *opaque) : _opaque(opaque) {}                          \
                                                                                 \
    public:                                                                      \
        ~DeviceImpl() = default;                                                 \
                                                                                 \
        template <typename... Args>                                              \
        static infiniStatus_t create(                                            \
            DeviceImpl **device_info,                                            \
            Args &&...args);                                                     \
                                                                                 \
        /* Invoke elementwise operation when all inputs have the same type */    \
        template <typename Op, typename Tdata, typename... Args>                 \
        void calculate(                                                          \
            const op::elementwise::ElementwiseInfo &info,                        \
            void *output,                                                        \
            const std::vector<const void *> &inputs,                             \
            Args &&...args);                                                     \
                                                                                 \
        /* Invoke elementwise operation for different input types */             \
        template <typename Op, typename Tout, typename... Tin,                   \
                  typename... Args,                                              \
                  std::enable_if_t<(sizeof...(Tin) == Op::num_inputs), int> = 0> \
        void calculate(                                                          \
            const op::elementwise::ElementwiseInfo &info,                        \
            void *output,                                                        \
            const std::vector<const void *> &inputs,                             \
            Args &&...args);                                                     \
    };                                                                           \
    }

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
              _info(info),                                                    \
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
    std::vector<bool> input_contiguous;
    std::vector<bool> input_broadcasted;
    std::vector<size_t> output_shape;
    std::vector<std::vector<size_t>> input_shapes;
    std::vector<ptrdiff_t> output_strides;
    std::vector<std::vector<ptrdiff_t>> input_strides;
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

    const size_t input_size = input_descs.size();
    const size_t out_ndim = output_desc->ndim();

    // Intializing the ElementwiseInfo struct
    info.output_size = output_desc->numel();
    info.ndim = out_ndim;
    info.output_contiguous = output_desc->isContiguous();

    for (const auto &desc : input_descs) {
        info.input_contiguous.emplace_back(desc->isContiguous());
    }

    for (size_t i = 0; i < input_size; ++i) {
        const auto &desc = input_descs[i];
        info.input_broadcasted.emplace_back(!info.input_contiguous[i] && (desc->ndim() != out_ndim || desc->hasBroadcastDim()));
    }

    info.output_shape = std::move(output_desc->shape());
    info.output_strides = std::move(output_desc->strides());
    for (const auto &desc : input_descs) {
        info.input_shapes.emplace_back(desc->shape());
        info.input_strides.emplace_back(desc->strides());
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::elementwise

#endif // __INFINIOP_ELEMENTWISE_H__
