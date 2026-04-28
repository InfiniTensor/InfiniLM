#ifndef __HISTC_NVIDIA_H__
#define __HISTC_NVIDIA_H__

#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../operator.h"
#include "../../../tensor.h"

namespace op::histc::nvidia {

class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    size_t _input_size;
    int64_t _bins;
    double _min_val;
    double _max_val;
    ptrdiff_t _input_stride;

    Descriptor(infiniDtype_t dtype, size_t input_size, int64_t bins,
               double min_val, double max_val, ptrdiff_t input_stride,
               infiniDevice_t device_type, int device_id)
        : InfiniopDescriptor{device_type, device_id},
          _dtype(dtype),
          _input_size(input_size),
          _bins(bins),
          _min_val(min_val),
          _max_val(max_val),
          _input_stride(input_stride) {}

public:
    ~Descriptor();

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t x_desc,
        int64_t bins,
        double min_val,
        double max_val);

    size_t workspaceSize() const { return 0; }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *y,
        const void *x,
        void *stream) const;
};

} // namespace op::histc::nvidia

#endif // __HISTC_NVIDIA_H__
