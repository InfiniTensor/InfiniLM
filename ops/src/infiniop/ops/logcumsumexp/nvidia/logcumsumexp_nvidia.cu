#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../handle.h"

#include "../cuda/kernel.cuh"
#include "logcumsumexp_nvidia.cuh"
#include <cstdint>

namespace op::logcumsumexp::nvidia {

// ============================================================
// Kernel Launch
// ============================================================

template <typename T>
void launch_kernel(
    void *y,
    const void *x,
    const LogCumSumExpInfo &info,
    void *stream) {

    auto x_ptr = reinterpret_cast<const T *>(x);
    auto y_ptr = reinterpret_cast<T *>(y);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    size_t outer = info.outer_size();
    size_t axis = info.axis_size();
    size_t inner = info.inner_size();

    size_t total = outer * inner;
    constexpr size_t block = 256;
    size_t grid = (total + block - 1) / block;

    op::logcumsumexp::cuda::logcumsumexp_kernel<T>
        <<<grid, block, 0, cuda_stream>>>(
            y_ptr,
            x_ptr,
            outer,
            axis,
            inner,

            info._x_axis_stride,
            info._x_inner_stride,
            info._x_outer_stride, // ✅ 新增

            info._y_axis_stride,
            info._y_inner_stride,
            info._y_outer_stride, // ✅ 新增

            info.exclusive(),
            info.reverse());
}

// ============================================================
// Descriptor
// ============================================================

struct Descriptor::Opaque {};

Descriptor::~Descriptor() {
    if (_opaque) {
        delete _opaque;
    }
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    int axis,
    int exclusive,
    int reverse) {

    auto info_result = LogCumSumExpInfo::create(y_desc, x_desc, axis, exclusive, reverse);
    if (!info_result) {
        return info_result.status();
    }

    *desc_ptr = new Descriptor(
        new Opaque(),
        info_result.take(),
        /*workspace*/ 0,
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void *stream) const {

    switch (_info.dtype()) {
    case INFINI_DTYPE_F16:
        launch_kernel<half>(y, x, _info, stream);
        break;
    case INFINI_DTYPE_BF16:
        launch_kernel<cuda_bfloat16>(y, x, _info, stream);
        break;
    case INFINI_DTYPE_F32:
        launch_kernel<float>(y, x, _info, stream);
        break;
    case INFINI_DTYPE_F64:
        launch_kernel<double>(y, x, _info, stream);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::logcumsumexp::nvidia
