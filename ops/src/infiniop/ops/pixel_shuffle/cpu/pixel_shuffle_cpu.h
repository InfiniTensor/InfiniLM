#ifndef __PIXEL_SHUFFLE_CPU_H__
#define __PIXEL_SHUFFLE_CPU_H__

#include "../../../devices/cpu/common_cpu.h"
#include "../../../operator.h"
#include <vector>

namespace op::pixel_shuffle::cpu {

struct PixelShuffleInfo {
    size_t batch;
    size_t in_channels;
    size_t out_channels;
    size_t height;
    size_t width;
    int upscale_factor;
    size_t input_size;
    size_t output_size;

    static utils::Result<PixelShuffleInfo> create(
        infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t y_desc,
        int upscale_factor);
};

class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    PixelShuffleInfo _info;

    Descriptor(infiniDtype_t dtype, PixelShuffleInfo info,
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
        int upscale_factor);

    size_t workspaceSize() const { return 0; }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *y,
        const void *x,
        void *stream) const;
};

} // namespace op::pixel_shuffle::cpu

#endif // __PIXEL_SHUFFLE_CPU_H__
