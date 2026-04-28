#ifndef __PAD_CPU_H__
#define __PAD_CPU_H__

#include "../../../devices/cpu/common_cpu.h"
#include "../../../operator.h"
#include <string>
#include <vector>

namespace op::pad::cpu {

enum class PadMode {
    CONSTANT,
    REFLECT,
    REPLICATE,
    CIRCULAR
};

struct PadInfo {
    size_t ndim;
    std::vector<size_t> input_shape;
    std::vector<ptrdiff_t> input_strides;
    std::vector<size_t> output_shape;
    std::vector<ptrdiff_t> output_strides;
    std::vector<int> pads; // [pad_left_dim0, pad_right_dim0, pad_left_dim1, pad_right_dim1, ...]
    PadMode mode;
    double value;

    static utils::Result<PadInfo> create(
        infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t y_desc,
        const void *pad,
        size_t pad_size,
        const char *mode_str,
        double value);
};

class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    PadInfo _info;

    Descriptor(infiniDtype_t dtype, PadInfo info,
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
        const void *pad,
        size_t pad_size,
        const char *mode,
        double value);

    size_t workspaceSize() const { return 0; }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *y,
        const void *x,
        void *stream) const;
};

} // namespace op::pad::cpu

#endif // __PAD_CPU_H__
