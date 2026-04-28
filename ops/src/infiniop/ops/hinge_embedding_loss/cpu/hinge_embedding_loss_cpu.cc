#include "hinge_embedding_loss_cpu.h"
#include "../../../../utils.h"
#include "../../../tensor.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

namespace op::hinge_embedding_loss::cpu {

utils::Result<HingeEmbeddingLossInfo> HingeEmbeddingLossInfo::create(
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t target_desc,
    infiniopTensorDescriptor_t y_desc,
    double margin,
    int reduction) {

    auto input_shape = input_desc->shape();
    auto target_shape = target_desc->shape();
    auto y_shape = y_desc->shape();

    if (input_shape != target_shape) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    Reduction red = static_cast<Reduction>(reduction);
    std::vector<size_t> expected_y_shape;
    if (red == Reduction::NONE) {
        expected_y_shape = input_shape;
    } else {
        // Mean or Sum: output is scalar
        expected_y_shape = {};
    }

    if (y_shape != expected_y_shape) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    HingeEmbeddingLossInfo info;
    info.ndim = input_shape.size();
    info.shape = input_shape;
    info.input_strides = input_desc->strides();
    info.target_strides = target_desc->strides();
    if (red == Reduction::NONE) {
        info.y_strides = y_desc->strides();
    }
    info.input_size = input_desc->numel();
    info.margin = margin;
    info.reduction = red;

    return utils::Result<HingeEmbeddingLossInfo>(std::move(info));
}

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t target_desc,
    double margin,
    int reduction) {

    auto dtype = input_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16);

    if (target_desc->dtype() != dtype || y_desc->dtype() != dtype) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    auto info_result = HingeEmbeddingLossInfo::create(input_desc, target_desc, y_desc, margin, reduction);
    CHECK_RESULT(info_result);

    *desc_ptr = new Descriptor(dtype, info_result.take(), handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename T>
void hinge_embedding_loss_impl(
    const HingeEmbeddingLossInfo &info,
    T *y,
    const T *input,
    const T *target) {

    size_t n = info.input_size;
    using Tcompute = std::conditional_t<std::is_same_v<T, double>, double, float>;
    Tcompute margin_val = static_cast<Tcompute>(info.margin);

    auto linearToOffset = [&](size_t linear, const std::vector<ptrdiff_t> &strides) -> ptrdiff_t {
        ptrdiff_t off = 0;
        for (size_t d = info.ndim; d-- > 0;) {
            size_t coord = (info.shape.empty() ? 0 : (linear % info.shape[d]));
            if (!info.shape.empty()) {
                linear /= info.shape[d];
            }
            off += static_cast<ptrdiff_t>(coord) * strides[d];
        }
        return off;
    };

    auto loss_value = [&](Tcompute in, Tcompute t) -> Tcompute {
        if (t == static_cast<Tcompute>(1)) {
            return in;
        }
        if (t == static_cast<Tcompute>(-1)) {
            return std::max(static_cast<Tcompute>(0), margin_val - in);
        }
        // PyTorch defines a fallback behavior for unexpected target values
        // (i.e., not exactly ±1): loss = max(input, margin).
        return std::max(in, margin_val);
    };

    if (info.reduction == Reduction::NONE) {
        // Element-wise loss
        for (size_t i = 0; i < n; ++i) {
            ptrdiff_t in_off = linearToOffset(i, info.input_strides);
            ptrdiff_t t_off = linearToOffset(i, info.target_strides);
            ptrdiff_t y_off = linearToOffset(i, info.y_strides);
            Tcompute t = utils::cast<Tcompute>(target[t_off]);
            Tcompute in = utils::cast<Tcompute>(input[in_off]);
            y[y_off] = utils::cast<T>(loss_value(in, t));
        }
    } else {
        // Sum or Mean
        Tcompute sum = static_cast<Tcompute>(0);
        for (size_t i = 0; i < n; ++i) {
            ptrdiff_t in_off = linearToOffset(i, info.input_strides);
            ptrdiff_t t_off = linearToOffset(i, info.target_strides);
            Tcompute t = utils::cast<Tcompute>(target[t_off]);
            Tcompute in = utils::cast<Tcompute>(input[in_off]);
            sum += loss_value(in, t);
        }
        if (info.reduction == Reduction::MEAN) {
            const Tcompute mean_val = (n > 0) ? (sum / static_cast<Tcompute>(n)) : std::numeric_limits<Tcompute>::quiet_NaN();
            y[0] = utils::cast<T>(mean_val);
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
    void *stream) const {

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        hinge_embedding_loss_impl<fp16_t>(_info, reinterpret_cast<fp16_t *>(y),
                                          reinterpret_cast<const fp16_t *>(input),
                                          reinterpret_cast<const fp16_t *>(target));
        break;
    case INFINI_DTYPE_BF16:
        hinge_embedding_loss_impl<bf16_t>(_info, reinterpret_cast<bf16_t *>(y),
                                          reinterpret_cast<const bf16_t *>(input),
                                          reinterpret_cast<const bf16_t *>(target));
        break;
    case INFINI_DTYPE_F32:
        hinge_embedding_loss_impl<float>(_info, reinterpret_cast<float *>(y),
                                         reinterpret_cast<const float *>(input),
                                         reinterpret_cast<const float *>(target));
        break;
    case INFINI_DTYPE_F64:
        hinge_embedding_loss_impl<double>(_info, reinterpret_cast<double *>(y),
                                          reinterpret_cast<const double *>(input),
                                          reinterpret_cast<const double *>(target));
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::hinge_embedding_loss::cpu
