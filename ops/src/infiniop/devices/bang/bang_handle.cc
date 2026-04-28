#include "../../tensor.h"
#include "common_bang.h"
#include "infiniop/tensor_descriptor.h"
#include <memory>
#include <vector>

namespace device::bang {

Handle::Handle(infiniDevice_t device, int device_id)
    : InfiniopHandle{device, device_id},
      _internal(std::make_shared<Handle::Internal>(device_id)) {}

auto Handle::internal() const -> const std::shared_ptr<Internal> & {
    return _internal;
}

Handle::Internal::Internal(int device_id) {
    cnrtDeviceGetAttribute(&_cluster_count, cnrtAttrClusterCount, device_id);
    cnrtDeviceGetAttribute(&_core_per_cluster, cnrtAttrMcorePerCluster, device_id);
}

infiniStatus_t Handle::Internal::useCnnl(cnrtQueue_t queue, const Fn<cnnlHandle_t> &f) const {
    auto handle = cnnl_handles.pop();
    if (!handle) {
        CHECK_BANG(cnnlCreate(&(*handle)));
    }
    CHECK_BANG(cnnlSetQueue(*handle, queue));
    CHECK_STATUS(f(*handle));
    cnnl_handles.push(std::move(*handle));
    return INFINI_STATUS_SUCCESS;
}

int Handle::Internal::getCorePerCluster() const { return _core_per_cluster; }
int Handle::Internal::getClusterCount() const { return _cluster_count; }

cnnlDataType_t getCnnlDtype(infiniDtype_t dt) {
    switch (dt) {
    case INFINI_DTYPE_F32:
        return CNNL_DTYPE_FLOAT;
    case INFINI_DTYPE_F64:
        return CNNL_DTYPE_DOUBLE;
    case INFINI_DTYPE_F16:
        return CNNL_DTYPE_HALF;
    case INFINI_DTYPE_I8:
        return CNNL_DTYPE_INT8;
    case INFINI_DTYPE_I32:
        return CNNL_DTYPE_INT32;
    case INFINI_DTYPE_U8:
        return CNNL_DTYPE_UINT8;
    case INFINI_DTYPE_BF16:
        return CNNL_DTYPE_BFLOAT16;
    case INFINI_DTYPE_I64:
        return CNNL_DTYPE_INT64;
    default:
        return CNNL_DTYPE_INVALID;
    }
}

infiniStatus_t setCnnlTensor(cnnlTensorDescriptor_t desc,
                             const InfiniopTensorDescriptor *layout) {
    std::vector<int> dims(layout->ndim());
    for (size_t i = 0; i < layout->ndim(); i++) {
        dims[i] = static_cast<int>(layout->shape()[i]);
    }
    CHECK_BANG(cnnlSetTensorDescriptor(desc, CNNL_LAYOUT_ARRAY,
                                       getCnnlDtype(layout->dtype()), dims.size(),
                                       dims.data()));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t setCnnlTensorEx(cnnlTensorDescriptor_t desc,
                               const InfiniopTensorDescriptor *layout) {
    std::vector<int> dim_size(layout->ndim()), dim_stride(layout->ndim());
    for (size_t i = 0; i < layout->ndim(); i++) {
        dim_size[i] = static_cast<int>(layout->shape()[i]);
        dim_stride[i] = static_cast<int>(layout->strides()[i]);
    }
    CHECK_BANG(cnnlSetTensorDescriptorEx(
        desc, CNNL_LAYOUT_ARRAY, getCnnlDtype(layout->dtype()),
        dim_size.size(), dim_size.data(), dim_stride.data()));
    return INFINI_STATUS_SUCCESS;
}

namespace cambricon {

Handle::Handle(int device_id)
    : bang::Handle(INFINI_DEVICE_CAMBRICON, device_id) {}

infiniStatus_t Handle::create(InfiniopHandle **handle_ptr, int device_id) {
    *handle_ptr = new Handle(device_id);
    return INFINI_STATUS_SUCCESS;
}

} // namespace cambricon

} // namespace device::bang
