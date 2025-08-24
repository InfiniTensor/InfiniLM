#ifndef __INFINIOP_ELEMENTWISE_H__
#define __INFINIOP_ELEMENTWISE_H__

#include "../../utils.h"
#include "../operator.h"
#include "../tensor.h"
#include <algorithm>
#include <array>
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
        size_t _workspace_size;                                               \
                                                                              \
        Descriptor(                                                           \
            infiniDtype_t dtype,                                              \
            op::elementwise::ElementwiseInfo info,                            \
            op::elementwise::NAMESPACE::DeviceImpl *device_info,              \
            size_t workspace_size,                                            \
            infiniDevice_t device_type,                                       \
            int device_id)                                                    \
            : InfiniopDescriptor{device_type, device_id},                     \
              _dtype(dtype),                                                  \
              _info(std::move(info)),                                         \
              _device_info(std::move(device_info)),                           \
              _workspace_size(workspace_size) {}                              \
                                                                              \
    public:                                                                   \
        ~Descriptor();                                                        \
                                                                              \
        size_t workspaceSize() const { return _workspace_size; }              \
                                                                              \
        static infiniStatus_t create(                                         \
            infiniopHandle_t handle,                                          \
            Descriptor **desc_ptr,                                            \
            infiniopTensorDescriptor_t output_desc,                           \
            std::vector<infiniopTensorDescriptor_t> input_descs);             \
                                                                              \
        infiniStatus_t calculate(                                             \
            void *workspace, size_t workspace_size,                           \
            void *output,                                                     \
            std::vector<const void *> inputs,                                 \
            void *stream) const;                                              \
    };                                                                        \
    }

namespace op::elementwise {

/**
 * @brief Stores the metadata required for performing an elementwise operation.
 *
 * This struct encapsulates shape, stride, and layout information for both
 * output and multiple input tensors involved in an elementwise operation.
 *
 * Memory is manually managed and freed in the destructor.
 * Supports move construction but disallows copy construction and copy/move assignment.
 *
 * Use ElementwiseInfo::create(...) to safely construct an instance from tensor descriptors.
 */
struct ElementwiseInfo {
private:
    std::vector<size_t> _meta;
    size_t _output_size;
    size_t _input_size;
    size_t _ndim;
    bool _output_contiguous;

    ElementwiseInfo(std::vector<size_t> meta,
                    size_t output_size,
                    size_t input_size,
                    size_t ndim,
                    bool output_contiguous)
        : _meta(std::move(meta)), _output_size(output_size),
          _input_size(input_size), _ndim(ndim),
          _output_contiguous(output_contiguous) {}

public:
    // Get the Memory size of the meta data in bytes
    inline size_t getMetaMemSize() const {
        return _meta.size() * sizeof(size_t);
    }
    inline const int8_t *getMetaStart() const {
        return reinterpret_cast<const int8_t *>(_meta.data());
    }
    inline size_t getOutputSize() const {
        return _output_size;
    }
    inline size_t getInputSize() const {
        return _input_size;
    }
    inline size_t getNdim() const {
        return _ndim;
    }
    inline bool isOutputContiguous() const {
        return _output_contiguous;
    }
    inline const size_t *getOutputShape() const {
        return reinterpret_cast<const size_t *>(_meta.data());
    }
    inline const ptrdiff_t *getOutputStrides() const {
        return reinterpret_cast<const ptrdiff_t *>(getOutputShape() + _ndim);
    }
    inline const size_t *getAllInputShapes() const {
        return reinterpret_cast<const size_t *>(getOutputStrides() + _ndim);
    }
    inline const size_t *getInputShape(const size_t &index) const {
        if (index < _input_size) {
            return reinterpret_cast<const size_t *>(getAllInputShapes() + index * _ndim);
        }
        return nullptr;
    }
    inline const ptrdiff_t *getAllInputStrides() const {
        return reinterpret_cast<const ptrdiff_t *>(getAllInputShapes() + _input_size * _ndim);
    }
    inline const ptrdiff_t *getInputStrides(const size_t &index) const {
        if (index < _input_size) {
            return reinterpret_cast<const ptrdiff_t *>(getAllInputStrides() + index * _ndim);
        }
        return nullptr;
    }
    inline const bool *getInputContiguous() const {
        return reinterpret_cast<const bool *>(getAllInputStrides() + _input_size * _ndim);
    }
    inline const bool *getInputBroadcasted() const {
        return reinterpret_cast<const bool *>(getInputContiguous() + _input_size);
    }

    using ResultType = utils::Result<ElementwiseInfo>;

    /**
     * @brief Construct ElementwiseInfo from output and input tensor descriptors.
     * @param output_desc Descriptor of the output tensor.
     * @param input_descs Descriptors of the input tensors.
     * @return Result<ElementwiseInfo> with the successfully constructed ElementwiseInfo,
     *         or the status code.
     */
    static ResultType create(
        infiniopTensorDescriptor_t output_desc,
        std::vector<infiniopTensorDescriptor_t> input_descs) {

        if (!output_desc || input_descs.empty()) {
            return INFINI_STATUS_BAD_PARAM;
        }

        // Destination cannot have broadcast setup
        if (output_desc->hasBroadcastDim()) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }

        auto input_size = input_descs.size();
        auto ndim = output_desc->ndim();
        auto output_size = output_desc->numel();
        auto output_contiguous = output_desc->isContiguous();

        // Allocate memory for meta
        auto shape_unit = output_desc->dim(0);
        auto stride_unit = output_desc->stride(0);
        size_t meta_mem_size = ndim * (sizeof(shape_unit) + sizeof(stride_unit))
                             + input_size * ndim * sizeof(shape_unit)
                             + input_size * ndim * sizeof(stride_unit)
                             + 2 * input_size * sizeof(bool);
        std::vector<size_t> meta(CEIL_DIV(meta_mem_size, sizeof(size_t)));
        int8_t *meta_ptr = reinterpret_cast<int8_t *>(meta.data());

        const auto output_shape = output_desc->shape();
        const auto output_strides = output_desc->strides();

        // Pointers to the sections within _meta
        size_t *output_shape_p = reinterpret_cast<size_t *>(meta_ptr);
        ptrdiff_t *output_strides_p = reinterpret_cast<ptrdiff_t *>(output_shape_p + ndim);
        size_t *input_shapes = reinterpret_cast<size_t *>(output_strides_p + ndim);
        ptrdiff_t *input_strides = reinterpret_cast<ptrdiff_t *>(input_shapes + input_size * ndim);
        bool *input_contiguous = reinterpret_cast<bool *>(input_strides + input_size * ndim);
        bool *input_broadcasted = input_contiguous + input_size;

        // Copy output shape and strides
        std::memcpy(output_shape_p, output_shape.data(), ndim * sizeof(*output_shape_p));
        std::memcpy(output_strides_p, output_strides.data(), ndim * sizeof(*output_strides_p));

        // Copy input shapes, strides, contiguous, and broadcasted flags
        for (size_t i = 0; i < input_size; ++i) {
            auto &desc = input_descs[i];
            const auto in_shape = desc->shape();
            const auto in_strides = desc->strides();
            std::memcpy(input_shapes + i * ndim, in_shape.data(), ndim * sizeof(*input_shapes));
            std::memcpy(input_strides + i * ndim, in_strides.data(), ndim * sizeof(*input_strides));
            input_contiguous[i] = desc->isContiguous();
            input_broadcasted[i] = !input_contiguous[i] && (desc->ndim() != ndim || desc->hasBroadcastDim());
        }

        ElementwiseInfo info(std::move(meta), output_size, input_size, ndim, output_contiguous);
        return ResultType(std::move(info));
    }
};
} // namespace op::elementwise

#endif // __INFINIOP_ELEMENTWISE_H__
