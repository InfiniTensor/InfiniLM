#include "logdet_cpu.h"
#include "../../../tensor.h"
#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>

namespace op::logdet::cpu {

utils::Result<LogdetInfo> LogdetInfo::create(
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t y_desc) {

    auto x_shape = x_desc->shape();
    auto y_shape = y_desc->shape();

    if (x_shape.size() != 2 || x_shape[0] != x_shape[1]) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    // Output is scalar
    if (y_shape.size() != 0 && (y_shape.size() != 1 || y_shape[0] != 1)) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    LogdetInfo info;
    info.matrix_size = x_shape[0];
    info.input_size = x_desc->numel();
    info.input_strides = x_desc->strides();

    return utils::Result<LogdetInfo>(std::move(info));
}

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc) {

    auto dtype = x_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F32, INFINI_DTYPE_F64);

    auto info_result = LogdetInfo::create(x_desc, y_desc);
    CHECK_RESULT(info_result);

    *desc_ptr = new Descriptor(dtype, info_result.take(), handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename T>
constexpr T singular_pivot_eps() {
    if constexpr (std::is_same_v<T, float>) {
        return static_cast<T>(1e-6f);
    }
    return static_cast<T>(1e-12);
}

template <typename T>
void logdet_impl(
    const LogdetInfo &info,
    T *y,
    const T *x,
    void *workspace) {

    const size_t n = info.matrix_size;
    T *U = reinterpret_cast<T *>(workspace);

    // Copy into a contiguous row-major buffer so the LU decomposition below can
    // use simple indexing, while still respecting arbitrary input strides.
    const ptrdiff_t s0 = info.input_strides[0];
    const ptrdiff_t s1 = info.input_strides[1];
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            U[i * n + j] = x[static_cast<ptrdiff_t>(i) * s0 + static_cast<ptrdiff_t>(j) * s1];
        }
    }

    int det_sign = 1;
    double log_abs_det = 0.0;

    for (size_t k = 0; k < n; ++k) {
        size_t pivot_row = k;
        double pivot_abs = std::abs(static_cast<double>(U[k * n + k]));
        for (size_t i = k + 1; i < n; ++i) {
            const double v = std::abs(static_cast<double>(U[i * n + k]));
            if (v > pivot_abs) {
                pivot_abs = v;
                pivot_row = i;
            }
        }

        if (pivot_abs <= static_cast<double>(singular_pivot_eps<T>())) {
            y[0] = utils::cast<T>(-std::numeric_limits<double>::infinity());
            return;
        }

        if (pivot_row != k) {
            for (size_t j = 0; j < n; ++j) {
                std::swap(U[k * n + j], U[pivot_row * n + j]);
            }
            det_sign *= -1;
        }

        const T pivot = U[k * n + k];
        if (pivot < static_cast<T>(0)) {
            det_sign *= -1;
        }
        log_abs_det += std::log(std::abs(static_cast<double>(pivot)));

        for (size_t i = k + 1; i < n; ++i) {
            const T factor = U[i * n + k] / pivot;
            U[i * n + k] = static_cast<T>(0);
            for (size_t j = k + 1; j < n; ++j) {
                U[i * n + j] -= factor * U[k * n + j];
            }
        }
    }

    if (det_sign <= 0) {
        y[0] = utils::cast<T>(std::numeric_limits<double>::quiet_NaN());
        return;
    }

    y[0] = utils::cast<T>(log_abs_det);
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void *stream) const {

    if (workspace_size < this->workspaceSize()) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    switch (_dtype) {
    case INFINI_DTYPE_F32:
        logdet_impl<float>(_info, reinterpret_cast<float *>(y), reinterpret_cast<const float *>(x), workspace);
        break;
    case INFINI_DTYPE_F64:
        logdet_impl<double>(_info, reinterpret_cast<double *>(y), reinterpret_cast<const double *>(x), workspace);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::logdet::cpu
