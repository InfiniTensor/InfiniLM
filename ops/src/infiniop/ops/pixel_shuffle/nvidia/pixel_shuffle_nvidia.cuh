#ifndef __PIXEL_SHUFFLE_NVIDIA_H__
#define __PIXEL_SHUFFLE_NVIDIA_H__

#include "../../../operator.h"
#include <array>
#include <cstddef>

namespace op::pixel_shuffle::nvidia {

class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    size_t batch;
    size_t in_channels;
    size_t out_channels;
    size_t height;
    size_t width;
    int upscale_factor;
    size_t input_size;
    size_t output_size;
    std::array<ptrdiff_t, 4> x_strides;
    std::array<ptrdiff_t, 4> y_strides;

    Descriptor(infiniDtype_t dtype, size_t batch, size_t in_channels, size_t out_channels,
               size_t height, size_t width, int upscale_factor,
               size_t input_size, size_t output_size,
               std::array<ptrdiff_t, 4> x_strides,
               std::array<ptrdiff_t, 4> y_strides,
               infiniDevice_t device_type, int device_id)
        : InfiniopDescriptor{device_type, device_id},
          _dtype(dtype),
          batch(batch),
          in_channels(in_channels),
          out_channels(out_channels),
          height(height),
          width(width),
          upscale_factor(upscale_factor),
          input_size(input_size),
          output_size(output_size),
          x_strides(x_strides),
          y_strides(y_strides) {}

public:
    ~Descriptor();

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t x_desc,
        int upscale_factor);

    size_t workspaceSize() const { return 0; }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *y,
        const void *x,
        void *stream) const;
};

} // namespace op::pixel_shuffle::nvidia

#endif // __PIXEL_SHUFFLE_NVIDIA_H__
