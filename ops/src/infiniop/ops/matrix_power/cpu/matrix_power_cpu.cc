#include "matrix_power_cpu.h"
#include "../../../tensor.h"
#include <algorithm>
#include <cstring>

namespace op::matrix_power::cpu {

utils::Result<MatrixPowerInfo> MatrixPowerInfo::create(
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t y_desc,
    int n) {

    auto x_shape = x_desc->shape();
    auto y_shape = y_desc->shape();

    if (x_shape.size() != 2 || x_shape[0] != x_shape[1]) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    if (y_shape != x_shape) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    if (n < 0) {
        return INFINI_STATUS_BAD_PARAM;
    }

    MatrixPowerInfo info;
    info.matrix_size = x_shape[0];
    info.n = static_cast<size_t>(n);
    info.input_size = x_desc->numel();
    info.output_size = y_desc->numel();

    return utils::Result<MatrixPowerInfo>(std::move(info));
}

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    int n) {

    auto dtype = x_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16);
    CHECK_OR_RETURN(y_desc->dtype() == dtype, INFINI_STATUS_BAD_TENSOR_DTYPE);

    CHECK_OR_RETURN(x_desc->isContiguous() && y_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
    CHECK_OR_RETURN(!x_desc->hasBroadcastDim() && !y_desc->hasBroadcastDim(), INFINI_STATUS_BAD_TENSOR_STRIDES);

    auto info_result = MatrixPowerInfo::create(x_desc, y_desc, n);
    CHECK_RESULT(info_result);

    *desc_ptr = new Descriptor(dtype, info_result.take(), handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename T>
void matrix_power_impl(
    const MatrixPowerInfo &info,
    T *y,
    const T *x,
    void *workspace) {

    size_t n = info.matrix_size;
    int power = static_cast<int>(info.n);

    // Use workspace for temporary matrices
    T *temp1 = reinterpret_cast<T *>(workspace);
    T *temp2 = temp1 + n * n;

    // Initialize result as identity matrix
    std::memset(y, 0, n * n * sizeof(T));
    for (size_t i = 0; i < n; ++i) {
        y[i * n + i] = utils::cast<T>(1.0);
    }

    // Copy input to temp1
    std::memcpy(temp1, x, n * n * sizeof(T));

    // Binary exponentiation
    while (power > 0) {
        if (power & 1) {
            // Multiply result by temp1
            std::memset(temp2, 0, n * n * sizeof(T));
            for (size_t i = 0; i < n; ++i) {
                for (size_t k = 0; k < n; ++k) {
                    T val = y[i * n + k];
                    for (size_t j = 0; j < n; ++j) {
                        temp2[i * n + j] += val * temp1[k * n + j];
                    }
                }
            }
            std::memcpy(y, temp2, n * n * sizeof(T));
        }
        // Square temp1
        std::memset(temp2, 0, n * n * sizeof(T));
        for (size_t i = 0; i < n; ++i) {
            for (size_t k = 0; k < n; ++k) {
                T val = temp1[i * n + k];
                for (size_t j = 0; j < n; ++j) {
                    temp2[i * n + j] += val * temp1[k * n + j];
                }
            }
        }
        std::memcpy(temp1, temp2, n * n * sizeof(T));
        power >>= 1;
    }
}

template <typename T>
void write_identity_impl(size_t n, T *y) {
    std::fill(y, y + n * n, utils::cast<T>(0.0));
    for (size_t i = 0; i < n; ++i) {
        y[i * n + i] = utils::cast<T>(1.0);
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void *stream) const {

    if (_info.matrix_size == 0) {
        return INFINI_STATUS_SUCCESS;
    }
    if (_info.n == 0) {
        const size_t n = _info.matrix_size;
        switch (_dtype) {
        case INFINI_DTYPE_F16:
            write_identity_impl<fp16_t>(n, reinterpret_cast<fp16_t *>(y));
            return INFINI_STATUS_SUCCESS;
        case INFINI_DTYPE_BF16:
            write_identity_impl<bf16_t>(n, reinterpret_cast<bf16_t *>(y));
            return INFINI_STATUS_SUCCESS;
        case INFINI_DTYPE_F32:
            write_identity_impl<float>(n, reinterpret_cast<float *>(y));
            return INFINI_STATUS_SUCCESS;
        case INFINI_DTYPE_F64:
            write_identity_impl<double>(n, reinterpret_cast<double *>(y));
            return INFINI_STATUS_SUCCESS;
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    }

    if (workspace_size < this->workspaceSize()) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    if (this->workspaceSize() > 0 && workspace == nullptr) {
        return INFINI_STATUS_BAD_PARAM;
    }

    switch (_dtype) {
    case INFINI_DTYPE_F16: {
        // Convert to float for computation
        std::vector<float> x_f(_info.input_size);
        std::vector<float> y_f(_info.output_size);
        float *workspace_f = reinterpret_cast<float *>(workspace);
        for (size_t i = 0; i < _info.input_size; ++i) {
            x_f[i] = utils::cast<float>(reinterpret_cast<const fp16_t *>(x)[i]);
        }
        MatrixPowerInfo info_f = _info;
        matrix_power_impl<float>(info_f, y_f.data(), x_f.data(), workspace_f);
        for (size_t i = 0; i < _info.output_size; ++i) {
            reinterpret_cast<fp16_t *>(y)[i] = utils::cast<fp16_t>(y_f[i]);
        }
        break;
    }
    case INFINI_DTYPE_BF16: {
        std::vector<float> x_f(_info.input_size);
        std::vector<float> y_f(_info.output_size);
        float *workspace_f = reinterpret_cast<float *>(workspace);
        for (size_t i = 0; i < _info.input_size; ++i) {
            x_f[i] = utils::cast<float>(reinterpret_cast<const bf16_t *>(x)[i]);
        }
        MatrixPowerInfo info_f = _info;
        matrix_power_impl<float>(info_f, y_f.data(), x_f.data(), workspace_f);
        for (size_t i = 0; i < _info.output_size; ++i) {
            reinterpret_cast<bf16_t *>(y)[i] = utils::cast<bf16_t>(y_f[i]);
        }
        break;
    }
    case INFINI_DTYPE_F32:
        matrix_power_impl<float>(_info, reinterpret_cast<float *>(y), reinterpret_cast<const float *>(x), workspace);
        break;
    case INFINI_DTYPE_F64:
        matrix_power_impl<double>(_info, reinterpret_cast<double *>(y), reinterpret_cast<const double *>(x), workspace);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::matrix_power::cpu
