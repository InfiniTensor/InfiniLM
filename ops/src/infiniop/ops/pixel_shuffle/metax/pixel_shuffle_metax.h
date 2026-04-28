#ifndef __PIXEL_SHUFFLE_METAX_H__
#define __PIXEL_SHUFFLE_METAX_H__

#include "../../../operator.h"

namespace op::pixel_shuffle::metax {

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

    Descriptor(infiniDtype_t dtype, size_t batch, size_t in_channels, size_t out_channels,
               size_t height, size_t width, int upscale_factor,
               size_t input_size, size_t output_size,
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
          output_size(output_size) {}

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

} // namespace op::pixel_shuffle::metax

#endif // __PIXEL_SHUFFLE_METAX_H__
