#ifndef __HARDTANH_CPU_H__
#define __HARDTANH_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include <algorithm>

namespace op::hardtanh::cpu {

class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    op::elementwise::ElementwiseInfo _info;
    size_t _workspace_size;
    float _min_val;
    float _max_val;

    Descriptor(infiniDtype_t dtype,
               op::elementwise::ElementwiseInfo info,
               size_t workspace_size,
               infiniDevice_t device_type,
               int device_id,
               float min_val,
               float max_val);

public:
    ~Descriptor();

    size_t workspaceSize() const { return _workspace_size; }

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t out_desc,
        std::vector<infiniopTensorDescriptor_t> input_desc_vec,
        float min_val,
        float max_val);

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *output,
        std::vector<const void *> inputs,
        void *stream) const;

    float minVal() const { return _min_val; }
    float maxVal() const { return _max_val; }
};

typedef struct HardTanhOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &x, float min_val, float max_val) const {
        T low = static_cast<T>(min_val);
        T high = static_cast<T>(max_val);
        T val = x < low ? low : x;
        return val > high ? high : val;
    }
} HardTanhOp;

} // namespace op::hardtanh::cpu

#endif
