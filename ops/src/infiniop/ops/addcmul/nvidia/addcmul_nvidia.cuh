#ifndef __ADDCMUL_NVIDIA_H__
#define __ADDCMUL_NVIDIA_H__

#include "../../../elementwise/nvidia/elementwise_nvidia_api.cuh"

namespace op::addcmul::nvidia {

// 为 addcmul 在 NVIDIA 端自定义 Descriptor，支持额外的标量参数 value
class Descriptor final : public InfiniopDescriptor {
    // 为保持与通用 Elementwise 框架的兼容，仍然保留这些成员
    infiniDtype_t _dtype;
    op::elementwise::ElementwiseInfo _info;
    std::unique_ptr<op::elementwise::nvidia::DeviceImpl> _device_info;
    size_t _workspace_size;
    float _value; // 标量系数 value

public:
    // 为自定义 CUDA kernel 记录张量元信息
    static constexpr int MAX_NDIM = 8;

    struct TensorMeta {
        int ndim;
        size_t shape[MAX_NDIM];
        ptrdiff_t strides[MAX_NDIM];
    };

    TensorMeta _out_meta{};
    TensorMeta _input_meta{};
    TensorMeta _t1_meta{};
    TensorMeta _t2_meta{};
    size_t _output_size{0};

    Descriptor(
        infiniDtype_t dtype,
        op::elementwise::ElementwiseInfo info,
        op::elementwise::nvidia::DeviceImpl *device_info,
        size_t workspace_size,
        infiniDevice_t device_type,
        int device_id)
        : InfiniopDescriptor{device_type, device_id},
          _dtype(dtype),
          _info(std::move(info)),
          _device_info(device_info),
          _workspace_size(workspace_size),
          _value(0.0f) {}

public:
    ~Descriptor();

    size_t workspaceSize() const { return _workspace_size; }

    // 额外接收 value 参数
    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t output_desc,
        std::vector<infiniopTensorDescriptor_t> input_descs,
        float value);

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *output,
        std::vector<const void *> inputs,
        void *stream) const;

    float getValue() const { return _value; }
};

} // namespace op::addcmul::nvidia

#endif // __ADDCMUL_NVIDIA_H__
