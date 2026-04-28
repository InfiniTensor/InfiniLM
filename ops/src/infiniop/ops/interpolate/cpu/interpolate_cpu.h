#ifndef __INTERPOLATE_CPU_H__
#define __INTERPOLATE_CPU_H__

#include "../../../operator.h"
#include "../../../tensor.h"
#include <string>
#include <vector>

namespace op::interpolate::cpu {

enum class InterpolateMode {
    NEAREST,
    LINEAR,
    BILINEAR,
    TRILINEAR,
    AREA
};

struct InterpolateInfo {
    size_t ndim;
    std::vector<size_t> input_shape;
    std::vector<size_t> output_shape;
    std::vector<ptrdiff_t> input_strides;
    std::vector<ptrdiff_t> output_strides;
    InterpolateMode mode;
    int align_corners;
    size_t input_size;
    size_t output_size;

    static utils::Result<InterpolateInfo> create(
        infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t y_desc,
        const char *mode_str,
        void *size,
        void *scale_factor,
        int align_corners);
};

class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    InterpolateInfo _info;

    Descriptor(infiniDtype_t dtype, InterpolateInfo info,
               infiniDevice_t device_type, int device_id)
        : InfiniopDescriptor{device_type, device_id},
          _dtype(dtype),
          _info(std::move(info)) {}

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

} // namespace op::interpolate::cpu

#endif // __INTERPOLATE_CPU_H__
