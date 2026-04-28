#include "dot_cpu.h"
#include "../../../../utils.h"

namespace op::dot::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {

    auto dtype = a_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16);

    // This op does not do implicit dtype conversion: y/a/b must match.
    if (b_desc->dtype() != dtype || y_desc->dtype() != dtype) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    // Check that y is a scalar (0D tensor or shape [1])
    auto y_shape = y_desc->shape();
    if (y_shape.size() != 0 && (y_shape.size() != 1 || y_shape[0] != 1)) {
        return INFINI_STATUS_BAD_PARAM;
    }

    // Check that a and b are 1D vectors with same length
    auto a_shape = a_desc->shape();
    auto b_shape = b_desc->shape();
    if (a_shape.size() != 1 || b_shape.size() != 1 || a_shape[0] != b_shape[0]) {
        return INFINI_STATUS_BAD_PARAM;
    }

    size_t n = a_shape[0];
    ptrdiff_t a_stride = a_desc->strides()[0];
    ptrdiff_t b_stride = b_desc->strides()[0];

    // Negative/broadcasted strides are not supported without an explicit base offset.
    if (a_stride <= 0 || b_stride <= 0) {
        return INFINI_STATUS_BAD_TENSOR_STRIDES;
    }

    *desc_ptr = new Descriptor(dtype, n, a_stride, b_stride, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *a,
    const void *b,
    void *stream) const {

    switch (_dtype) {
    case INFINI_DTYPE_F16: {
        const fp16_t *a_ptr = reinterpret_cast<const fp16_t *>(a);
        const fp16_t *b_ptr = reinterpret_cast<const fp16_t *>(b);
        float result = 0.0f;
        for (size_t i = 0; i < _n; ++i) {
            result += utils::cast<float>(a_ptr[i * _a_stride]) * utils::cast<float>(b_ptr[i * _b_stride]);
        }
        *reinterpret_cast<fp16_t *>(y) = utils::cast<fp16_t>(result);
        break;
    }
    case INFINI_DTYPE_BF16: {
        const bf16_t *a_ptr = reinterpret_cast<const bf16_t *>(a);
        const bf16_t *b_ptr = reinterpret_cast<const bf16_t *>(b);
        float result = 0.0f;
        for (size_t i = 0; i < _n; ++i) {
            result += utils::cast<float>(a_ptr[i * _a_stride]) * utils::cast<float>(b_ptr[i * _b_stride]);
        }
        *reinterpret_cast<bf16_t *>(y) = utils::cast<bf16_t>(result);
        break;
    }
    case INFINI_DTYPE_F32: {
        const float *a_ptr = reinterpret_cast<const float *>(a);
        const float *b_ptr = reinterpret_cast<const float *>(b);
        float result = 0.0f;
        for (size_t i = 0; i < _n; ++i) {
            result += a_ptr[i * _a_stride] * b_ptr[i * _b_stride];
        }
        *reinterpret_cast<float *>(y) = result;
        break;
    }
    case INFINI_DTYPE_F64: {
        const double *a_ptr = reinterpret_cast<const double *>(a);
        const double *b_ptr = reinterpret_cast<const double *>(b);
        double result = 0.0;
        for (size_t i = 0; i < _n; ++i) {
            result += a_ptr[i * _a_stride] * b_ptr[i * _b_stride];
        }
        *reinterpret_cast<double *>(y) = result;
        break;
    }
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::dot::cpu
