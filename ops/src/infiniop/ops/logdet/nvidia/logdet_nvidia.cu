#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../tensor.h"
#include "logdet_nvidia.cuh"
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace op::logdet::nvidia {

Descriptor::~Descriptor() = default;

template <typename T>
__global__ void pack_matrix_kernel(
    T *dst,
    const T *src,
    ptrdiff_t s0,
    ptrdiff_t s1,
    size_t n) {

    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = n * n;
    if (idx >= total) {
        return;
    }
    const size_t i = idx / n;
    const size_t j = idx % n;
    dst[idx] = src[static_cast<ptrdiff_t>(i) * s0 + static_cast<ptrdiff_t>(j) * s1];
}

template <typename T>
__global__ void logdet_lu_kernel(
    T *packed,
    size_t n,
    T *out) {

    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    int det_sign = 1;
    double log_abs_det = 0.0;
    const double eps = std::is_same_v<T, float> ? 1e-6 : 1e-12;

    for (size_t k = 0; k < n; ++k) {
        size_t pivot_row = k;
        double pivot_abs = fabs(static_cast<double>(packed[k * n + k]));
        for (size_t i = k + 1; i < n; ++i) {
            const double v = fabs(static_cast<double>(packed[i * n + k]));
            if (v > pivot_abs) {
                pivot_abs = v;
                pivot_row = i;
            }
        }

        if (pivot_abs <= eps) {
            *out = -std::numeric_limits<T>::infinity();
            return;
        }

        if (pivot_row != k) {
            for (size_t j = 0; j < n; ++j) {
                const T tmp = packed[k * n + j];
                packed[k * n + j] = packed[pivot_row * n + j];
                packed[pivot_row * n + j] = tmp;
            }
            det_sign *= -1;
        }

        const T pivot = packed[k * n + k];
        if (pivot < static_cast<T>(0)) {
            det_sign *= -1;
        }
        log_abs_det += log(fabs(static_cast<double>(pivot)));

        for (size_t i = k + 1; i < n; ++i) {
            const T factor = packed[i * n + k] / pivot;
            packed[i * n + k] = static_cast<T>(0);
            for (size_t j = k + 1; j < n; ++j) {
                packed[i * n + j] -= factor * packed[k * n + j];
            }
        }
    }

    if (det_sign <= 0) {
        *out = static_cast<T>(std::numeric_limits<double>::quiet_NaN());
        return;
    }
    *out = static_cast<T>(log_abs_det);
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc) {

    auto dtype = x_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F32, INFINI_DTYPE_F64);

    auto x_shape = x_desc->shape();
    auto y_shape = y_desc->shape();

    if (x_shape.size() != 2 || x_shape[0] != x_shape[1]) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    if (y_shape.size() != 0 && (y_shape.size() != 1 || y_shape[0] != 1)) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    *desc_ptr = new Descriptor(dtype, x_shape[0], x_desc->numel(), x_desc->strides(),
                               handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
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

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    if (_dtype == INFINI_DTYPE_F32) {
        using T = float;
        T *packed = reinterpret_cast<T *>(workspace);
        const ptrdiff_t s0 = input_strides[0];
        const ptrdiff_t s1 = input_strides[1];
        constexpr int BLOCK_SIZE = 256;
        const int blocks = static_cast<int>((input_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
        pack_matrix_kernel<T><<<blocks, BLOCK_SIZE, 0, cuda_stream>>>(
            packed, reinterpret_cast<const T *>(x), s0, s1, matrix_size);
        logdet_lu_kernel<T><<<1, 1, 0, cuda_stream>>>(
            packed, matrix_size, reinterpret_cast<T *>(y));
        return INFINI_STATUS_SUCCESS;
    }

    {
        using T = double;
        T *packed = reinterpret_cast<T *>(workspace);
        const ptrdiff_t s0 = input_strides[0];
        const ptrdiff_t s1 = input_strides[1];
        constexpr int BLOCK_SIZE = 256;
        const int blocks = static_cast<int>((input_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
        pack_matrix_kernel<T><<<blocks, BLOCK_SIZE, 0, cuda_stream>>>(
            packed, reinterpret_cast<const T *>(x), s0, s1, matrix_size);
        logdet_lu_kernel<T><<<1, 1, 0, cuda_stream>>>(
            packed, matrix_size, reinterpret_cast<T *>(y));
        return INFINI_STATUS_SUCCESS;
    }
}

} // namespace op::logdet::nvidia
