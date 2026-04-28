#ifndef __ADDCMUL_CPU_H__
#define __ADDCMUL_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include <cmath>
#include <type_traits>

namespace op::addcmul::cpu {

struct AddcmulOp {
public:
    // addcmul 是三元算子: out = input + value * t1 * t2
    static constexpr size_t num_inputs = 3;

    template <typename T, typename Scalar>
    T operator()(const T &input, const T &t1, const T &t2, Scalar value) const {
        // 对于 float, double 等原生浮点类型
        if constexpr (std::is_floating_point_v<T>) {
            return input + static_cast<T>(value) * t1 * t2;
        } else {
            // 对于 fp16, bf16 等类型，提升至 float 计算以保证精度并处理标量乘法
            float f_input = static_cast<float>(input);
            float f_t1 = static_cast<float>(t1);
            float f_t2 = static_cast<float>(t2);
            float v = static_cast<float>(value);
            return static_cast<T>(f_input + v * f_t1 * f_t2);
        }
    }
};

// 为 addcmul 在 CPU 端自定义 Descriptor，支持额外的标量参数 value
class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    op::elementwise::ElementwiseInfo _info;
    std::unique_ptr<op::elementwise::cpu::DeviceImpl> _device_info;
    size_t _workspace_size;
    float _value; // 标量系数 value

    Descriptor(
        infiniDtype_t dtype,
        op::elementwise::ElementwiseInfo info,
        op::elementwise::cpu::DeviceImpl *device_info,
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

} // namespace op::addcmul::cpu

#endif // __ADDCMUL_CPU_H__
