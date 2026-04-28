#include "../../../devices/moore/moore_handle.h"
#include "logcumsumexp_moore.h"
#include "logcumsumexp_moore_kernel.h"
#include <cstdint>
#include <musa_fp16.h>
#include <musa_runtime.h>

namespace op::logcumsumexp::moore {

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
    auto musa_stream = reinterpret_cast<musaStream_t>(stream);

    size_t outer = info.outer_size();
    size_t axis = info.axis_size();
    size_t inner = info.inner_size();

    size_t total = outer * inner;
    constexpr size_t block = 256;
    size_t grid = (total + block - 1) / block;

    op::logcumsumexp::moore::logcumsumexp_kernel<T>
        <<<grid, block, 0, musa_stream>>>(
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
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    int axis,
    int exclusive,
    int reverse) {

    auto handle = reinterpret_cast<device::moore::Handle *>(handle_);

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
        // Moore 架构通常使用 __mt_bfloat16
        launch_kernel<__mt_bfloat16>(y, x, _info, stream);
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

} // namespace op::logcumsumexp::moore
