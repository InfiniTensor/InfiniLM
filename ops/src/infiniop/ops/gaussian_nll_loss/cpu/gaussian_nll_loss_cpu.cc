#include "gaussian_nll_loss_cpu.h"
#include "../../../../utils.h"
#include <cmath>

namespace op::gaussian_nll_loss::cpu {

utils::Result<GaussianNllLossInfo> GaussianNllLossInfo::create(
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t target_desc,
    infiniopTensorDescriptor_t var_desc,
    infiniopTensorDescriptor_t y_desc,
    int full,
    double eps,
    int reduction) {

    auto input_shape = input_desc->shape();
    auto target_shape = target_desc->shape();
    auto var_shape = var_desc->shape();
    auto y_shape = y_desc->shape();

    if (input_shape != target_shape || input_shape != var_shape) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    Reduction red = static_cast<Reduction>(reduction);
    std::vector<size_t> expected_y_shape;
    if (red == Reduction::NONE) {
        expected_y_shape = input_shape;
    } else {
        expected_y_shape = {};
    }

    if (y_shape != expected_y_shape) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    GaussianNllLossInfo info;
    info.input_size = input_desc->numel();
    info.full = full;
    info.eps = eps;
    info.reduction = red;

    return utils::Result<GaussianNllLossInfo>(std::move(info));
}

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t target_desc,
    infiniopTensorDescriptor_t var_desc,
    int full,
    double eps,
    int reduction) {

    auto dtype = input_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16);

    auto info_result = GaussianNllLossInfo::create(input_desc, target_desc, var_desc, y_desc, full, eps, reduction);
    CHECK_RESULT(info_result);

    *desc_ptr = new Descriptor(dtype, info_result.take(), handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename T>
void gaussian_nll_loss_impl(
    const GaussianNllLossInfo &info,
    T *y,
    const T *input,
    const T *target,
    const T *var) {

    size_t n = info.input_size;
    const double eps_val = info.eps;
    const double log_2pi = std::log(2.0 * 3.14159265358979323846);

    if (info.reduction == Reduction::NONE) {
        // Element-wise loss
        for (size_t i = 0; i < n; ++i) {
            const double diff = utils::cast<double>(input[i]) - utils::cast<double>(target[i]);
            double var_val = utils::cast<double>(var[i]);
            if (var_val < eps_val) {
                var_val = eps_val;
            }
            double loss = 0.5 * (std::log(var_val) + (diff * diff) / var_val);
            if (info.full) {
                loss += 0.5 * log_2pi;
            }
            y[i] = utils::cast<T>(loss);
        }
    } else {
        // Sum or Mean
        double sum = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const double diff = utils::cast<double>(input[i]) - utils::cast<double>(target[i]);
            double var_val = utils::cast<double>(var[i]);
            if (var_val < eps_val) {
                var_val = eps_val;
            }
            double loss = 0.5 * (std::log(var_val) + (diff * diff) / var_val);
            if (info.full) {
                loss += 0.5 * log_2pi;
            }
            sum += loss;
        }
        if (info.reduction == Reduction::MEAN) {
            y[0] = utils::cast<T>(sum / static_cast<double>(n));
        } else {
            y[0] = utils::cast<T>(sum);
        }
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *input,
    const void *target,
    const void *var,
    void *stream) const {

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        gaussian_nll_loss_impl<fp16_t>(_info, reinterpret_cast<fp16_t *>(y),
                                       reinterpret_cast<const fp16_t *>(input),
                                       reinterpret_cast<const fp16_t *>(target),
                                       reinterpret_cast<const fp16_t *>(var));
        break;
    case INFINI_DTYPE_BF16:
        gaussian_nll_loss_impl<bf16_t>(_info, reinterpret_cast<bf16_t *>(y),
                                       reinterpret_cast<const bf16_t *>(input),
                                       reinterpret_cast<const bf16_t *>(target),
                                       reinterpret_cast<const bf16_t *>(var));
        break;
    case INFINI_DTYPE_F32:
        gaussian_nll_loss_impl<float>(_info, reinterpret_cast<float *>(y),
                                      reinterpret_cast<const float *>(input),
                                      reinterpret_cast<const float *>(target),
                                      reinterpret_cast<const float *>(var));
        break;
    case INFINI_DTYPE_F64:
        gaussian_nll_loss_impl<double>(_info, reinterpret_cast<double *>(y),
                                       reinterpret_cast<const double *>(input),
                                       reinterpret_cast<const double *>(target),
                                       reinterpret_cast<const double *>(var));
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::gaussian_nll_loss::cpu
