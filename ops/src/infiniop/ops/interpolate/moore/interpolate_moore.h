#ifndef __INTERPOLATE_MOORE_H__
#define __INTERPOLATE_MOORE_H__

#include "../../../operator.h"
#include <vector>

namespace op::interpolate::moore {

enum class InterpolateMode {
    NEAREST,
    LINEAR,
    BILINEAR,
    TRILINEAR,
    AREA
};

class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    size_t ndim;
    std::vector<size_t> input_shape;
    std::vector<size_t> output_shape;
    std::vector<ptrdiff_t> input_strides;
    std::vector<ptrdiff_t> output_strides;
    InterpolateMode mode;
    int align_corners;
    size_t input_size;
    size_t output_size;

    Descriptor(infiniDtype_t dtype, size_t ndim,
               std::vector<size_t> input_shape, std::vector<size_t> output_shape,
               std::vector<ptrdiff_t> input_strides, std::vector<ptrdiff_t> output_strides,
               InterpolateMode mode, int align_corners,
               size_t input_size, size_t output_size,
               infiniDevice_t device_type, int device_id)
        : InfiniopDescriptor{device_type, device_id},
          _dtype(dtype),
          ndim(ndim),
          input_shape(std::move(input_shape)),
          output_shape(std::move(output_shape)),
          input_strides(std::move(input_strides)),
          output_strides(std::move(output_strides)),
          mode(mode),
          align_corners(align_corners),
          input_size(input_size),
          output_size(output_size) {}

public:
    ~Descriptor();

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t x_desc,
        const char *mode,
        void *size,
        void *scale_factor,
        int align_corners);

    size_t workspaceSize() const { return 0; }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *y,
        const void *x,
        void *stream) const;
};
} // namespace op::interpolate::moore

#endif // __INTERPOLATE_MOORE_H__
