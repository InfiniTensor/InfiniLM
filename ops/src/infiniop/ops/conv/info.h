#ifndef __CONV_INFO_H__
#define __CONV_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"

#ifdef ENABLE_CUDA_API
#include "../../devices/nvidia/nvidia_handle.cuh"
#endif

namespace op::conv {
class ConvInfo;
} // namespace op::conv

namespace op::conv {

class ConvInfo {
private:
    std::vector<size_t> _meta;
    size_t _ndim;
    size_t _batch;
    size_t _in_channels;
    size_t _out_channels;
    size_t _spatial_sizes;
    size_t _bias_dims_size;
    size_t _padded_shape_size;

    ConvInfo(std::vector<size_t> meta,
             size_t ndim,
             size_t batch,
             size_t in_channels,
             size_t out_channels,
             size_t spatial_sizes,
             size_t bias_dims_size,
             size_t padded_shape_size)
        : _meta(std::move(meta)),
          _ndim(ndim),
          _batch(batch),
          _in_channels(in_channels),
          _out_channels(out_channels),
          _spatial_sizes(spatial_sizes),
          _bias_dims_size(bias_dims_size),
          _padded_shape_size(padded_shape_size) {}

public:
    inline size_t ndim() const { return _ndim; }
    inline size_t batch() const { return _batch; }
    inline size_t in_channels() const { return _in_channels; }
    inline size_t out_channels() const { return _out_channels; }
    inline size_t spatial_sizes() const { return _spatial_sizes; }
    inline size_t bias_dims_size() const { return _bias_dims_size; }
    inline size_t padded_shape_size() const { return _padded_shape_size; }

    inline size_t getMetaMemSize() const {
        return _meta.size() * sizeof(size_t);
    }
    inline const int8_t *getMetaStart() const {
        return reinterpret_cast<const int8_t *>(_meta.data());
    }

    inline const size_t *getInputDims() const {
        return _meta.data();
    }
    inline const size_t *getKernelDims() const {
        return getInputDims() + _ndim;
    }
    inline const size_t *getOutputDims() const {
        return getKernelDims() + _ndim;
    }
    inline const size_t *getBiasDims() const {
        return getOutputDims() + _ndim;
    }
    inline const size_t *getPadsInfo() const {
        return getBiasDims() + _bias_dims_size;
    }
    inline const ptrdiff_t *getStridesInfo() const {
        return reinterpret_cast<const ptrdiff_t *>(getPadsInfo()) + _ndim;
    }
    inline const size_t *getDilationsInfo() const {
        return reinterpret_cast<const size_t *>(getStridesInfo()) + _ndim;
    }
    inline const size_t *getPaddedShape() const {
        return getDilationsInfo() + _ndim;
    }

    inline size_t input_dim(size_t i) const {
        return i < _ndim ? getInputDims()[i] : 0;
    }
    inline size_t kernel_dim(size_t i) const {
        return i < _ndim ? getKernelDims()[i] : 0;
    }
    inline size_t output_dim(size_t i) const {
        return i < _ndim ? getOutputDims()[i] : 0;
    }
    inline size_t bias_dim(size_t i) const {
        return i < _bias_dims_size ? getBiasDims()[i] : 0;
    }
    inline size_t pad_info(size_t i) const {
        return i < _ndim ? getPadsInfo()[i] : 0;
    }
    inline ptrdiff_t stride_info(size_t i) const {
        return i < _ndim ? getStridesInfo()[i] : 0;
    }
    inline size_t dilation_info(size_t i) const {
        return i < _ndim ? getDilationsInfo()[i] : 0;
    }
    inline size_t padded_shape_dim(size_t i) const {
        return i < _padded_shape_size ? getPaddedShape()[i] : 0;
    }

    static utils::Result<ConvInfo> create(
        infiniopHandle_t handle_,
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t w_desc,
        infiniopTensorDescriptor_t b_desc,
        const void *pads,
        const void *strides,
        const void *dilations,
        size_t n);
};

inline utils::Result<size_t> calculateConvOutputSize(
    size_t input_size,
    size_t kernel_size,
    size_t padding,
    size_t stride,
    size_t dilation) {
    if (stride == 0) {
        return utils::Result<size_t>(INFINI_STATUS_BAD_TENSOR_SHAPE);
    }
    if (dilation == 0) {
        return utils::Result<size_t>(INFINI_STATUS_BAD_TENSOR_SHAPE);
    }
    if (kernel_size == 0) {
        return utils::Result<size_t>(INFINI_STATUS_BAD_TENSOR_SHAPE);
    }
    size_t effective_kernel = dilation * (kernel_size - 1) + 1;

    size_t padded_input = input_size + 2 * padding;

    if (padded_input < effective_kernel) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    size_t output_size = (padded_input - effective_kernel) / stride + 1;

    return utils::Result<size_t>(output_size);
}

inline utils::Result<ConvInfo> ConvInfo::create(
    infiniopHandle_t handle_,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc,
    infiniopTensorDescriptor_t b_desc,
    const void *pads,
    const void *strides,
    const void *dilations,
    size_t n) {

    auto dtype = y_desc->dtype();
    if (dtype != x_desc->dtype() || dtype != w_desc->dtype()) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    size_t ndim = n;
    size_t new_dims = n + 2;

    if (x_desc->ndim() < new_dims || y_desc->ndim() < new_dims || w_desc->ndim() < new_dims) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    size_t batch = x_desc->shape()[0];
    size_t in_channels = x_desc->shape()[1];
    size_t out_channels = w_desc->shape()[0];

    if (y_desc->shape()[0] != batch || y_desc->shape()[1] != out_channels || w_desc->shape()[1] != in_channels) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    size_t bias_dims_size = (b_desc != nullptr) ? x_desc->ndim() : 0;

    const size_t *pads_ptr = reinterpret_cast<const size_t *>(pads);
    bool has_padding = false;
    if (pads_ptr != nullptr) {
        for (size_t i = 0; i < ndim; ++i) {
            if (pads_ptr[i] > 0) {
                has_padding = true;
                break;
            }
        }
    }
    size_t padded_shape_size = has_padding ? (ndim + 2) : 0;

    // 计算meta总大小
    size_t meta_size = ndim * 6 + bias_dims_size + padded_shape_size;
    std::vector<size_t> meta(meta_size);

    size_t *input_dims = meta.data();
    size_t *kernel_dims = input_dims + ndim;
    size_t *output_dims = kernel_dims + ndim;
    size_t *bias_dims = output_dims + ndim;
    size_t *pads_info = bias_dims + bias_dims_size;
    ptrdiff_t *strides_info = reinterpret_cast<ptrdiff_t *>(pads_info) + ndim;
    size_t *dilations_info = reinterpret_cast<size_t *>(strides_info) + ndim;
    size_t *padded_shape = dilations_info + ndim;

    const ptrdiff_t *strides_ptr = reinterpret_cast<const ptrdiff_t *>(strides);
    const size_t *dilations_ptr = reinterpret_cast<const size_t *>(dilations);

    size_t spatial_sizes = 1;

    for (size_t i = 0; i < ndim; i++) {
        input_dims[i] = x_desc->shape()[i + 2];
        kernel_dims[i] = w_desc->shape()[i + 2];
        output_dims[i] = y_desc->shape()[i + 2];
        pads_info[i] = pads_ptr == nullptr ? 0 : pads_ptr[i];
        strides_info[i] = strides_ptr == nullptr ? 1 : strides_ptr[i];
        dilations_info[i] = dilations_ptr == nullptr ? 1 : dilations_ptr[i];
        spatial_sizes = spatial_sizes * output_dims[i];
        auto output_result = calculateConvOutputSize(
            input_dims[i],
            kernel_dims[i],
            pads_info[i],
            strides_info[i],
            dilations_info[i]);
        CHECK_RESULT(output_result);
        size_t expected_output = output_result.take();
        if (output_dims[i] != expected_output) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
    }

    if (bias_dims_size > 0) {
        std::fill(bias_dims, bias_dims + bias_dims_size, 1);
        bias_dims[1] = b_desc->shape()[0];
    }

    if (padded_shape_size > 0) {
        padded_shape[0] = batch;
        padded_shape[1] = in_channels;
        for (size_t i = 0; i < ndim; ++i) {
            padded_shape[i + 2] = input_dims[i] + 2 * pads_info[i];
        }
    }

    ConvInfo info(std::move(meta), ndim, batch, in_channels, out_channels,
                  spatial_sizes, bias_dims_size, padded_shape_size);

    return utils::Result<ConvInfo>(info);
}

} // namespace op::conv

#endif // __CONV_INFO_H__
