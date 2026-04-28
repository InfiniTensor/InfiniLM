#ifndef __HARDTANH_MOORE_API_H__
#define __HARDTANH_MOORE_API_H__

#include "../../../elementwise/moore/elementwise_moore_api.h"

namespace op::hardtanh::moore {

class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    op::elementwise::ElementwiseInfo _info;
    std::unique_ptr<op::elementwise::moore::DeviceImpl> _device_info;
    size_t _workspace_size;
    float _min_val;
    float _max_val;

    Descriptor(infiniDtype_t dtype,
               op::elementwise::ElementwiseInfo info,
               op::elementwise::moore::DeviceImpl *device_info,
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

} // namespace op::hardtanh::moore

#endif // __HARDTANH_MOORE_API_H__
