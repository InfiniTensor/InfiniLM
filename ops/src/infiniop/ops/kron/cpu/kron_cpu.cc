#include "kron_cpu.h"
#include "../../../../utils.h"
#include "../../../tensor.h"
#include <type_traits>

namespace op::kron::cpu {

utils::Result<KronInfo> KronInfo::create(
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t y_desc) {

    auto a_shape = a_desc->shape();
    auto b_shape = b_desc->shape();
    auto y_shape = y_desc->shape();

    // Kron requires same number of dimensions
    if (a_shape.size() != b_shape.size()) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    size_t ndim = a_shape.size();

    // Output shape: each dimension is a_shape[i] * b_shape[i]
    std::vector<size_t> expected_y_shape(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        expected_y_shape[i] = a_shape[i] * b_shape[i];
    }

    if (y_shape != expected_y_shape) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    KronInfo info;
    info.ndim = ndim;
    info.a_shape = a_shape;
    info.b_shape = b_shape;
    info.y_shape = y_shape;
    info.a_strides = a_desc->strides();
    info.b_strides = b_desc->strides();
    info.y_strides = y_desc->strides();
    info.a_size = a_desc->numel();
    info.b_size = b_desc->numel();
    info.y_size = y_desc->numel();

    return utils::Result<KronInfo>(std::move(info));
}

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {

    auto dtype = a_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16);

    if (b_desc->dtype() != dtype || y_desc->dtype() != dtype) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    auto info_result = KronInfo::create(a_desc, b_desc, y_desc);
    CHECK_RESULT(info_result);

    *desc_ptr = new Descriptor(dtype, info_result.take(), handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename T>
void kron_impl(
    const KronInfo &info,
    T *y,
    const T *a,
    const T *b) {

    // Kronecker product: for each element a[i] in A, multiply it with entire B
    // y[output_idx] = a[a_idx] * b[b_idx]
    // where output coordinates are computed from a and b coordinates

    // Iterate over output tensor
    std::vector<size_t> y_coords(info.ndim, 0);
    for (size_t y_idx = 0; y_idx < info.y_size; ++y_idx) {
        // Convert linear index to coordinates
        size_t temp = y_idx;
        for (size_t d = info.ndim; d-- > 0;) {
            y_coords[d] = temp % info.y_shape[d];
            temp /= info.y_shape[d];
        }

        // Compute corresponding a and b coordinates, then use descriptor strides
        // to compute element offsets (supports non-contiguous / broadcasted strides).
        ptrdiff_t a_off = 0;
        ptrdiff_t b_off = 0;
        ptrdiff_t out_off = 0;
        for (size_t d = 0; d < info.ndim; ++d) {
            size_t a_coord = y_coords[d] / info.b_shape[d];
            size_t b_coord = y_coords[d] % info.b_shape[d];
            a_off += static_cast<ptrdiff_t>(a_coord) * info.a_strides[d];
            b_off += static_cast<ptrdiff_t>(b_coord) * info.b_strides[d];
            out_off += static_cast<ptrdiff_t>(y_coords[d]) * info.y_strides[d];
        }

        if constexpr (std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t>) {
            float av = utils::cast<float>(a[a_off]);
            float bv = utils::cast<float>(b[b_off]);
            y[out_off] = utils::cast<T>(av * bv);
        } else {
            y[out_off] = a[a_off] * b[b_off];
        }
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *a,
    const void *b,
    void *stream) const {

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        kron_impl<fp16_t>(_info, reinterpret_cast<fp16_t *>(y),
                          reinterpret_cast<const fp16_t *>(a),
                          reinterpret_cast<const fp16_t *>(b));
        break;
    case INFINI_DTYPE_BF16:
        kron_impl<bf16_t>(_info, reinterpret_cast<bf16_t *>(y),
                          reinterpret_cast<const bf16_t *>(a),
                          reinterpret_cast<const bf16_t *>(b));
        break;
    case INFINI_DTYPE_F32:
        kron_impl<float>(_info, reinterpret_cast<float *>(y),
                         reinterpret_cast<const float *>(a),
                         reinterpret_cast<const float *>(b));
        break;
    case INFINI_DTYPE_F64:
        kron_impl<double>(_info, reinterpret_cast<double *>(y),
                          reinterpret_cast<const double *>(a),
                          reinterpret_cast<const double *>(b));
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::kron::cpu
