#include "rms_norm_kunlun.h"
#include "../../../devices/kunlun/kunlun_common.h"
#include <memory>
#include <stdint.h>

void rmsNormF32(void *y, long stride_y, const void *x, long stride_x, const void *w, int m, int n, float epsilon, XPUStream stream);

namespace op::rms_norm::kunlun {

struct Descriptor::Opaque {
    std::shared_ptr<device::kunlun::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc,
    float epsilon) {
    auto result = RMSNormInfo::create(y_desc, x_desc, w_desc, epsilon);
    CHECK_RESULT(result);

    auto info = result.take();

    if (info.x_strides[1] != 1 || info.y_strides[1] != 1) {
        return INFINI_STATUS_BAD_TENSOR_STRIDES;
    }

    if (info.ndim() != 2) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    *desc_ptr = new Descriptor(
        new Descriptor::Opaque{static_cast<device::kunlun::Handle *>(handle)->internal()},
        info,
        0,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t launchKernel(
    int m, int n,
    void *y, infiniDtype_t atype, ptrdiff_t stride_y,
    const void *x, ptrdiff_t stride_x,
    const void *w, infiniDtype_t wtype,
    float epsilon,
    kunlunStream_t stream) {

    if (atype == INFINI_DTYPE_F32 && wtype == INFINI_DTYPE_F32) {
        rmsNormF32(y, static_cast<long>(stride_y), x, static_cast<long>(stride_x), w, m, n, epsilon, stream);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size,
                                     void *y, const void *x, const void *w, void *stream) const {

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    auto stride_x = _info.x_strides[0];
    auto stride_y = _info.y_strides[0];
    int n = static_cast<int>(_info.dim());
    int m = static_cast<int>(_info.shape[0]);

    launchKernel(m, n, y, _info.atype, stride_y, x, stride_x, w, _info.wtype, _info.epsilon, reinterpret_cast<kunlunStream_t>(stream));
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::rms_norm::kunlun
