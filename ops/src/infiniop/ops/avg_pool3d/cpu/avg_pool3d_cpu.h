#ifndef __AVG_POOL3D_CPU_H__
#define __AVG_POOL3D_CPU_H__

#include "../../../devices/cpu/common_cpu.h"
#include "../../../operator.h"
#include "../../../tensor.h"
#include <vector>

namespace op::avg_pool3d::cpu {

struct AvgPool3dInfo {
    size_t batch;
    size_t channels;
    size_t input_d, input_h, input_w;
    size_t output_d, output_h, output_w;
    size_t kernel_d, kernel_h, kernel_w;
    size_t stride_d, stride_h, stride_w;
    size_t pad_d, pad_h, pad_w;
    std::vector<ptrdiff_t> input_strides;
    std::vector<ptrdiff_t> output_strides;

    static utils::Result<AvgPool3dInfo> create(
        infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t y_desc,
        void *kernel_size,
        void *stride,
        void *padding);
};

class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    AvgPool3dInfo _info;

    Descriptor(infiniDtype_t dtype, AvgPool3dInfo info,
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
        void *kernel_size,
        void *stride,
        void *padding);

    size_t workspaceSize() const { return 0; }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *y,
        const void *x,
        void *stream) const;
};

} // namespace op::avg_pool3d::cpu

#endif // __AVG_POOL3D_CPU_H__
