#ifndef __SOFTPLUS_KUNLUN_H__
#define __SOFTPLUS_KUNLUN_H__

#include "../../../elementwise/kunlun/elementwise_kunlun_api.h"

namespace op::softplus::kunlun {
class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    op::elementwise::ElementwiseInfo _info;
    std::unique_ptr<op::elementwise::kunlun::DeviceImpl> _device_info;
    size_t _workspace_size;
    float _beta, _threshold;
    Descriptor(
        infiniDtype_t dtype,
        op::elementwise::ElementwiseInfo info,
        op::elementwise::kunlun::DeviceImpl *device_info,
        size_t workspace_size, infiniDevice_t device_type,
        int device_id,
        float beta, float threshold)
        : InfiniopDescriptor{device_type, device_id},
          _dtype(dtype), _info(std::move(info)),
          _device_info(std::move(device_info)),
          _workspace_size(workspace_size),
          _beta(beta),
          _threshold(threshold) {}

public:
    ~Descriptor();
    size_t workspaceSize() const { return _workspace_size; }
    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t output_desc,
        std::vector<infiniopTensorDescriptor_t> input_descs,
        float beta, float threshold);

    infiniStatus_t calculate(void *workspace, size_t workspace_size, void *output, std::vector<const void *> inputs, void *stream) const;
};
} // namespace op::softplus::kunlun

#endif
