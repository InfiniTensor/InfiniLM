#include "diff_cpu.h"
#include "../../../tensor.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace op::diff::cpu {

utils::Result<DiffInfo> DiffInfo::create(
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t y_desc,
    int dim,
    int n) {

    if (n <= 0) {
        return INFINI_STATUS_BAD_PARAM;
    }

    auto x_shape = x_desc->shape();
    auto y_shape = y_desc->shape();
    size_t ndim = x_desc->ndim();

    if (dim < 0) {
        dim += static_cast<int>(ndim);
    }
    if (dim < 0 || dim >= static_cast<int>(ndim)) {
        return INFINI_STATUS_BAD_PARAM;
    }

    if (x_shape[dim] <= static_cast<size_t>(n)) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    // Calculate output shape
    std::vector<size_t> expected_output_shape = x_shape;
    expected_output_shape[dim] -= n;

    if (y_shape != expected_output_shape) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    DiffInfo info;
    info.ndim = ndim;
    info.dim = dim;
    info.n = n;
    info.input_shape = x_shape;
    info.output_shape = y_shape;
    info.input_strides = x_desc->strides();
    info.output_strides = y_desc->strides();
    info.input_size = x_desc->numel();
    info.output_size = y_desc->numel();

    return utils::Result<DiffInfo>(std::move(info));
}

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    int dim,
    int n) {

    auto dtype = x_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16);

    auto info_result = DiffInfo::create(x_desc, y_desc, dim, n);
    CHECK_RESULT(info_result);

    *desc_ptr = new Descriptor(dtype, info_result.take(), handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename T>
void diff_impl(
    const DiffInfo &info,
    T *y,
    const T *x) {

    // n-th order forward difference along `dim`:
    //   y[i] = sum_{k=0..n} (-1)^(n-k) * C(n,k) * x[i+k]
    // Implemented directly to:
    // - avoid intermediate buffers (and their size pitfalls for n>1)
    // - respect input/output strides (tests cover as_strided cases)

    auto binom = [](int n, int k) -> double {
        if (k < 0 || k > n) {
            return 0.0;
        }
        k = std::min(k, n - k);
        double res = 1.0;
        for (int i = 1; i <= k; ++i) {
            res *= static_cast<double>(n - (k - i));
            res /= static_cast<double>(i);
        }
        return res;
    };

    std::vector<double> coeff(static_cast<size_t>(info.n) + 1);
    for (int k = 0; k <= info.n; ++k) {
        double c = binom(info.n, k);
        if (((info.n - k) & 1) != 0) {
            c = -c;
        }
        coeff[static_cast<size_t>(k)] = c;
    }

    const auto &out_shape = info.output_shape;
    const auto &in_strides = info.input_strides;
    const auto &out_strides = info.output_strides;
    const size_t out_numel = info.output_size;
    const ptrdiff_t stride_dim = in_strides[static_cast<size_t>(info.dim)];

    auto unravel_index = [](size_t linear, const std::vector<size_t> &shape, std::vector<size_t> &idx) {
        const size_t ndim = shape.size();
        for (size_t d = ndim; d-- > 0;) {
            const size_t s = shape[d];
            idx[d] = linear % s;
            linear /= s;
        }
    };

#pragma omp parallel
    {
        std::vector<size_t> idx(info.ndim, 0);

#pragma omp for
        for (ptrdiff_t linear = 0; linear < static_cast<ptrdiff_t>(out_numel); ++linear) {
            unravel_index(static_cast<size_t>(linear), out_shape, idx);

            ptrdiff_t y_off = 0;
            ptrdiff_t x_base_off = 0;
            for (size_t d = 0; d < info.ndim; ++d) {
                y_off += static_cast<ptrdiff_t>(idx[d]) * out_strides[d];
                x_base_off += static_cast<ptrdiff_t>(idx[d]) * in_strides[d];
            }

            double acc = 0.0;
            for (int k = 0; k <= info.n; ++k) {
                const ptrdiff_t x_off = x_base_off + static_cast<ptrdiff_t>(k) * stride_dim;
                acc += coeff[static_cast<size_t>(k)] * utils::cast<double>(x[x_off]);
            }

            y[y_off] = utils::cast<T>(acc);
        }
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void *stream) const {

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        diff_impl<fp16_t>(_info, reinterpret_cast<fp16_t *>(y), reinterpret_cast<const fp16_t *>(x));
        break;
    case INFINI_DTYPE_BF16:
        diff_impl<bf16_t>(_info, reinterpret_cast<bf16_t *>(y), reinterpret_cast<const bf16_t *>(x));
        break;
    case INFINI_DTYPE_F32:
        diff_impl<float>(_info, reinterpret_cast<float *>(y), reinterpret_cast<const float *>(x));
        break;
    case INFINI_DTYPE_F64:
        diff_impl<double>(_info, reinterpret_cast<double *>(y), reinterpret_cast<const double *>(x));
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::diff::cpu
