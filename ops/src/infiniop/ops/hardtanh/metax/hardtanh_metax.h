#ifndef __HARDTANH_METAX_API_H__
#define __HARDTANH_METAX_API_H__

#include "../../../elementwise/metax/elementwise_metax_api.h"

namespace op::hardtanh::metax {

class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    op::elementwise::ElementwiseInfo _info;
    std::unique_ptr<op::elementwise::metax::DeviceImpl> _device_info;
    size_t _workspace_size;
    float _min_val;
    float _max_val;

    Descriptor(infiniDtype_t dtype,
               op::elementwise::ElementwiseInfo info,
               op::elementwise::metax::DeviceImpl *device_info,
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
};

} // namespace op::hardtanh::metax

#endif // __HARDTANH_METAX_API_H__
