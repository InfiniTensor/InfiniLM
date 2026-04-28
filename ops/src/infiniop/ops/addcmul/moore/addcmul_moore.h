#ifndef __ADDCMUL_MOORE_H__
#define __ADDCMUL_MOORE_H__

// 1. 切换到 Moore 平台的 Elementwise API
#include "../../../elementwise/moore/elementwise_moore_api.h"

namespace op::addcmul::moore {

/**
 * 为 addcmul 在 Moore 端自定义 Descriptor
 * 保持与 NVIDIA 版本一致的结构，以便于跨平台对齐
 */
class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    op::elementwise::ElementwiseInfo _info;
    // 2. 切换到 Moore 设备的实现指针
    std::unique_ptr<op::elementwise::moore::DeviceImpl> _device_info;
    size_t _workspace_size;
    float _value; // 标量系数 value

public:
    // 摩尔线程 MUSA 同样支持 stride 访问，记录张量元信息
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
        op::elementwise::moore::DeviceImpl *device_info, // 3. 修改构造函数参数类型
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

    // 4. 保持相同的接口，接收 value 参数
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

} // namespace op::addcmul::moore

#endif // __ADDCMUL_MOORE_H__
