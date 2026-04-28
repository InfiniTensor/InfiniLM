#include "pixel_shuffle_cpu.h"
#include "../../../tensor.h"

namespace op::pixel_shuffle::cpu {

utils::Result<PixelShuffleInfo> PixelShuffleInfo::create(
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t y_desc,
    int upscale_factor) {

    if (upscale_factor <= 0) {
        return INFINI_STATUS_BAD_PARAM;
    }

    auto x_shape = x_desc->shape();
    auto y_shape = y_desc->shape();

    if (x_shape.size() != 4) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    size_t batch = x_shape[0];
    size_t in_channels = x_shape[1];
    size_t height = x_shape[2];
    size_t width = x_shape[3];

    // Input: (N, C*r^2, H, W) -> Output: (N, C, H*r, W*r)
    if (in_channels % (upscale_factor * upscale_factor) != 0) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    size_t out_channels = in_channels / (upscale_factor * upscale_factor);
    size_t out_height = height * upscale_factor;
    size_t out_width = width * upscale_factor;

    std::vector<size_t> expected_y_shape = {batch, out_channels, out_height, out_width};
    if (y_shape != expected_y_shape) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    PixelShuffleInfo info;
    info.batch = batch;
    info.in_channels = in_channels;
    info.out_channels = out_channels;
    info.height = height;
    info.width = width;
    info.upscale_factor = upscale_factor;
    info.input_size = x_desc->numel();
    info.output_size = y_desc->numel();

    return utils::Result<PixelShuffleInfo>(std::move(info));
}

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    int upscale_factor) {

    auto dtype = x_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16);
    CHECK_OR_RETURN(y_desc->dtype() == dtype, INFINI_STATUS_BAD_TENSOR_DTYPE);

    CHECK_OR_RETURN(x_desc->isContiguous() && y_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
    CHECK_OR_RETURN(!x_desc->hasBroadcastDim() && !y_desc->hasBroadcastDim(), INFINI_STATUS_BAD_TENSOR_STRIDES);

    auto info_result = PixelShuffleInfo::create(x_desc, y_desc, upscale_factor);
    CHECK_RESULT(info_result);

    *desc_ptr = new Descriptor(dtype, info_result.take(), handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename T>
void pixel_shuffle_impl(
    const PixelShuffleInfo &info,
    T *y,
    const T *x) {

    int r = info.upscale_factor;

    // Input: (N, C*r^2, H, W)
    // Output: (N, C, H*r, W*r)
    for (size_t n = 0; n < info.batch; ++n) {
        for (size_t c = 0; c < info.out_channels; ++c) {
            for (size_t h = 0; h < info.height; ++h) {
                for (size_t w = 0; w < info.width; ++w) {
                    for (int i = 0; i < r; ++i) {
                        for (int j = 0; j < r; ++j) {
                            // Input channel index
                            size_t in_c = c * r * r + i * r + j;
                            // Input position
                            size_t in_idx = ((n * info.in_channels + in_c) * info.height + h) * info.width + w;
                            // Output position
                            size_t out_h = h * r + i;
                            size_t out_w = w * r + j;
                            size_t out_idx = ((n * info.out_channels + c) * (info.height * r) + out_h) * (info.width * r) + out_w;
                            y[out_idx] = x[in_idx];
                        }
                    }
                }
            }
        }
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void *stream) const {

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        pixel_shuffle_impl<fp16_t>(_info, reinterpret_cast<fp16_t *>(y), reinterpret_cast<const fp16_t *>(x));
        break;
    case INFINI_DTYPE_BF16:
        pixel_shuffle_impl<bf16_t>(_info, reinterpret_cast<bf16_t *>(y), reinterpret_cast<const bf16_t *>(x));
        break;
    case INFINI_DTYPE_F32:
        pixel_shuffle_impl<float>(_info, reinterpret_cast<float *>(y), reinterpret_cast<const float *>(x));
        break;
    case INFINI_DTYPE_F64:
        pixel_shuffle_impl<double>(_info, reinterpret_cast<double *>(y), reinterpret_cast<const double *>(x));
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::pixel_shuffle::cpu
