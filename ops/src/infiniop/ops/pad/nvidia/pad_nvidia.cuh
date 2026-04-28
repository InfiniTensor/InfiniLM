#ifndef __PAD_NVIDIA_CUH__
#define __PAD_NVIDIA_CUH__

#include "../../../operator.h"

#include <vector>

namespace op::pad::nvidia {

enum class PadMode : int {
    CONSTANT = 0,
    REFLECT = 1,
    REPLICATE = 2,
    CIRCULAR = 3,
};

class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    size_t _ndim;
    PadMode _mode;
    double _value;

    std::vector<size_t> _input_shape;
    std::vector<ptrdiff_t> _input_strides;
    std::vector<size_t> _output_shape;
    std::vector<ptrdiff_t> _output_strides;
    std::vector<int> _pads; // [pad_left_dim0, pad_right_dim0, ...] in logical dim order

    size_t _output_numel;

    Descriptor(
        infiniDtype_t dtype,
        size_t ndim,
        PadMode mode,
        double value,
        std::vector<size_t> input_shape,
        std::vector<ptrdiff_t> input_strides,
        std::vector<size_t> output_shape,
        std::vector<ptrdiff_t> output_strides,
        std::vector<int> pads,
        size_t output_numel,
        infiniDevice_t device_type,
        int device_id)
        : InfiniopDescriptor{device_type, device_id},
          _dtype(dtype),
          _ndim(ndim),
          _mode(mode),
          _value(value),
          _input_shape(std::move(input_shape)),
          _input_strides(std::move(input_strides)),
          _output_shape(std::move(output_shape)),
          _output_strides(std::move(output_strides)),
          _pads(std::move(pads)),
          _output_numel(output_numel) {}

public:
    ~Descriptor();

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t x_desc,
        const void *pad,
        size_t pad_size,
        const char *mode,
        double value);

    size_t workspaceSize() const;

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *y,
        const void *x,
        void *stream) const;
};

} // namespace op::pad::nvidia

#endif // __PAD_NVIDIA_CUH__
