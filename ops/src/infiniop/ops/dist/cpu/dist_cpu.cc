#include "dist_cpu.h"
#include "../../../tensor.h"
#include <algorithm>
#include <cmath>

namespace op::dist::cpu {

utils::Result<DistInfo> DistInfo::create(
    infiniopTensorDescriptor_t x1_desc,
    infiniopTensorDescriptor_t x2_desc,
    infiniopTensorDescriptor_t y_desc,
    double p) {

    auto x1_shape = x1_desc->shape();
    auto x2_shape = x2_desc->shape();
    auto y_shape = y_desc->shape();

    // Check that x1 and x2 have same shape
    if (x1_shape != x2_shape) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    // Check that y is a scalar (0D tensor or shape [1])
    if (y_shape.size() != 0 && (y_shape.size() != 1 || y_shape[0] != 1)) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    DistInfo info;
    info.input_size = x1_desc->numel();
    info.p = p;
    info.x1_strides = x1_desc->strides();
    info.x2_strides = x2_desc->strides();
    info.shape = x1_shape;
    info.ndim = x1_desc->ndim();

    return utils::Result<DistInfo>(std::move(info));
}

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x1_desc,
    infiniopTensorDescriptor_t x2_desc,
    double p) {

    auto dtype = x1_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16);

    auto info_result = DistInfo::create(x1_desc, x2_desc, y_desc, p);
    CHECK_RESULT(info_result);

    *desc_ptr = new Descriptor(dtype, info_result.take(), handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename T>
void dist_impl(
    const DistInfo &info,
    T *y,
    const T *x1,
    const T *x2) {

    double sum = 0.0;
    const double p = info.p;

    for (size_t i = 0; i < info.input_size; ++i) {
        size_t idx1 = info.x1_strides.size() == 1 && info.x1_strides[0] == 1
                        ? i
                        : op::common_cpu::indexToOffset(i, info.ndim, info.shape.data(), info.x1_strides.data());
        size_t idx2 = info.x2_strides.size() == 1 && info.x2_strides[0] == 1
                        ? i
                        : op::common_cpu::indexToOffset(i, info.ndim, info.shape.data(), info.x2_strides.data());

        double diff = utils::cast<double>(x1[idx1]) - utils::cast<double>(x2[idx2]);
        double abs_diff = std::abs(diff);

        if (p == 0.0) {
            // L0 norm: count non-zero differences
            if (abs_diff > 1e-10) {
                sum += 1.0;
            }
        } else if (p == std::numeric_limits<double>::infinity()) {
            // L-infinity norm: max absolute difference
            sum = std::max(sum, abs_diff);
        } else {
            // Lp norm: sum of |diff|^p
            sum += std::pow(abs_diff, p);
        }
    }

    // Take p-th root (except for p=0 and p=inf)
    if (p == 0.0) {
        *y = utils::cast<T>(sum);
    } else if (p == std::numeric_limits<double>::infinity()) {
        *y = utils::cast<T>(sum);
    } else {
        *y = utils::cast<T>(std::pow(sum, 1.0 / p));
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x1,
    const void *x2,
    void *stream) const {

    switch (_dtype) {
    case INFINI_DTYPE_F16: {
        dist_impl<fp16_t>(_info, reinterpret_cast<fp16_t *>(y),
                          reinterpret_cast<const fp16_t *>(x1),
                          reinterpret_cast<const fp16_t *>(x2));
        break;
    }
    case INFINI_DTYPE_BF16: {
        dist_impl<bf16_t>(_info, reinterpret_cast<bf16_t *>(y),
                          reinterpret_cast<const bf16_t *>(x1),
                          reinterpret_cast<const bf16_t *>(x2));
        break;
    }
    case INFINI_DTYPE_F32: {
        dist_impl<float>(_info, reinterpret_cast<float *>(y),
                         reinterpret_cast<const float *>(x1),
                         reinterpret_cast<const float *>(x2));
        break;
    }
    case INFINI_DTYPE_F64: {
        dist_impl<double>(_info, reinterpret_cast<double *>(y),
                          reinterpret_cast<const double *>(x1),
                          reinterpret_cast<const double *>(x2));
        break;
    }
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::dist::cpu
