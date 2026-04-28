#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../handle.h"
#include "../../../tensor.h"
#include "diff_nvidia.cuh"
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <type_traits>

namespace op::diff::nvidia {

Descriptor::~Descriptor() = default;

template <typename T>
__device__ __forceinline__ T from_f32(float v);

template <>
__device__ __forceinline__ half from_f32<half>(float v) {
    return __float2half(v);
}

template <>
__device__ __forceinline__ cuda_bfloat16 from_f32<cuda_bfloat16>(float v) {
    return __float2bfloat16_rn(v);
}

template <>
__device__ __forceinline__ float from_f32<float>(float v) {
    return v;
}

struct Diff1Indexing {
    static constexpr int kMaxNdim = 8;

    int ndim;
    int dim;
    int64_t out_shape[kMaxNdim];
    int64_t in_strides[kMaxNdim];
    int64_t out_strides[kMaxNdim];
};

template <typename T>
__global__ void diff1_strided_kernel(
    T *out,
    const T *in,
    size_t out_numel,
    Diff1Indexing indexing) {

    const size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= out_numel) {
        return;
    }

    int64_t idx[Diff1Indexing::kMaxNdim] = {0};
    size_t tmp = linear;
    for (int d = indexing.ndim - 1; d >= 0; --d) {
        const int64_t s = indexing.out_shape[d];
        idx[d] = static_cast<int64_t>(tmp % static_cast<size_t>(s));
        tmp /= static_cast<size_t>(s);
    }

    int64_t y_off = 0;
    int64_t x_base_off = 0;
    for (int d = 0; d < indexing.ndim; ++d) {
        y_off += idx[d] * indexing.out_strides[d];
        x_base_off += idx[d] * indexing.in_strides[d];
    }

    const int64_t stride_dim = indexing.in_strides[indexing.dim];
    const int64_t x_off1 = x_base_off;
    const int64_t x_off2 = x_base_off + stride_dim;

    if constexpr (std::is_same_v<T, double>) {
        out[y_off] = in[x_off2] - in[x_off1];
    } else {
        float a;
        float b;
        if constexpr (std::is_same_v<T, half>) {
            a = __half2float(in[x_off1]);
            b = __half2float(in[x_off2]);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            a = __bfloat162float(in[x_off1]);
            b = __bfloat162float(in[x_off2]);
        } else { // float
            a = static_cast<float>(in[x_off1]);
            b = static_cast<float>(in[x_off2]);
        }
        out[y_off] = from_f32<T>(b - a);
    }
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    int dim,
    int n) {

    auto dtype = x_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16);

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

    std::vector<size_t> expected_output_shape = x_shape;
    expected_output_shape[dim] -= n;

    if (y_shape != expected_output_shape) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    *desc_ptr = new Descriptor(dtype, ndim, dim, n, x_shape, y_shape,
                               x_desc->strides(), y_desc->strides(),
                               x_desc->numel(), y_desc->numel(),
                               handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void *stream) const {

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    constexpr int BLOCK_SIZE = 256;

    auto numel_of = [](const std::vector<size_t> &shape) -> size_t {
        return std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies<size_t>{});
    };
    auto contiguous_strides = [](const std::vector<size_t> &shape) -> std::vector<ptrdiff_t> {
        std::vector<ptrdiff_t> strides(shape.size(), 1);
        ptrdiff_t running = 1;
        for (size_t d = shape.size(); d-- > 0;) {
            strides[d] = running;
            running *= static_cast<ptrdiff_t>(shape[d]);
        }
        return strides;
    };
    auto fill_indexing = [&](Diff1Indexing &indexing,
                             const std::vector<size_t> &out_shape,
                             const std::vector<ptrdiff_t> &in_strides,
                             const std::vector<ptrdiff_t> &out_strides) -> infiniStatus_t {
        indexing.ndim = static_cast<int>(_ndim);
        indexing.dim = _dim;
        if (indexing.ndim > Diff1Indexing::kMaxNdim) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        for (int d = 0; d < Diff1Indexing::kMaxNdim; ++d) {
            indexing.out_shape[d] = 1;
            indexing.in_strides[d] = 0;
            indexing.out_strides[d] = 0;
        }
        for (size_t d = 0; d < _ndim; ++d) {
            indexing.out_shape[d] = static_cast<int64_t>(out_shape[d]);
            indexing.in_strides[d] = static_cast<int64_t>(in_strides[d]);
            indexing.out_strides[d] = static_cast<int64_t>(out_strides[d]);
        }
        return INFINI_STATUS_SUCCESS;
    };

    auto launch_diff1 = [&](void *out_ptr,
                            const void *in_ptr,
                            const std::vector<size_t> &out_shape,
                            const std::vector<ptrdiff_t> &in_strides,
                            const std::vector<ptrdiff_t> &out_strides) -> infiniStatus_t {
        const size_t out_numel = numel_of(out_shape);
        const int blocks = static_cast<int>((out_numel + BLOCK_SIZE - 1) / BLOCK_SIZE);
        Diff1Indexing indexing{};
        auto st = fill_indexing(indexing, out_shape, in_strides, out_strides);
        if (st != INFINI_STATUS_SUCCESS) {
            return st;
        }

        switch (_dtype) {
        case INFINI_DTYPE_F16:
            diff1_strided_kernel<half><<<blocks, BLOCK_SIZE, 0, cuda_stream>>>(
                reinterpret_cast<half *>(out_ptr), reinterpret_cast<const half *>(in_ptr), out_numel, indexing);
            return INFINI_STATUS_SUCCESS;
        case INFINI_DTYPE_BF16:
            diff1_strided_kernel<cuda_bfloat16><<<blocks, BLOCK_SIZE, 0, cuda_stream>>>(
                reinterpret_cast<cuda_bfloat16 *>(out_ptr), reinterpret_cast<const cuda_bfloat16 *>(in_ptr), out_numel, indexing);
            return INFINI_STATUS_SUCCESS;
        case INFINI_DTYPE_F32:
            diff1_strided_kernel<float><<<blocks, BLOCK_SIZE, 0, cuda_stream>>>(
                reinterpret_cast<float *>(out_ptr), reinterpret_cast<const float *>(in_ptr), out_numel, indexing);
            return INFINI_STATUS_SUCCESS;
        case INFINI_DTYPE_F64:
            diff1_strided_kernel<double><<<blocks, BLOCK_SIZE, 0, cuda_stream>>>(
                reinterpret_cast<double *>(out_ptr), reinterpret_cast<const double *>(in_ptr), out_numel, indexing);
            return INFINI_STATUS_SUCCESS;
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    };

    if (_n == 1) {
        return launch_diff1(y, x, _output_shape, _input_strides, _output_strides);
    }

    if (workspace_size < this->workspaceSize()) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    const size_t elem_size = infiniSizeOf(_dtype);
    const size_t dim_size = _input_shape[static_cast<size_t>(_dim)];
    const size_t outer = _input_size / dim_size;
    const size_t max_intermediate = outer * (dim_size - 1);

    auto *ws = reinterpret_cast<uint8_t *>(workspace);
    void *buf_a = ws;
    void *buf_b = ws + max_intermediate * elem_size;

    std::vector<size_t> current_shape = _input_shape;
    std::vector<ptrdiff_t> current_in_strides = _input_strides;

    std::vector<size_t> out_shape = current_shape;
    out_shape[static_cast<size_t>(_dim)] -= 1;
    std::vector<ptrdiff_t> out_strides = contiguous_strides(out_shape);

    auto st = launch_diff1(buf_a, x, out_shape, current_in_strides, out_strides);
    if (st != INFINI_STATUS_SUCCESS) {
        return st;
    }

    current_shape = out_shape;
    current_in_strides = out_strides;
    bool a_is_input = true;

    for (int stage = 1; stage < _n - 1; ++stage) {
        out_shape = current_shape;
        out_shape[static_cast<size_t>(_dim)] -= 1;
        out_strides = contiguous_strides(out_shape);

        void *in_buf = a_is_input ? buf_a : buf_b;
        void *out_buf = a_is_input ? buf_b : buf_a;
        st = launch_diff1(out_buf, in_buf, out_shape, current_in_strides, out_strides);
        if (st != INFINI_STATUS_SUCCESS) {
            return st;
        }
        current_shape = out_shape;
        current_in_strides = out_strides;
        a_is_input = !a_is_input;
    }

    void *in_buf = a_is_input ? buf_a : buf_b;
    return launch_diff1(y, in_buf, _output_shape, current_in_strides, _output_strides);
}

} // namespace op::diff::nvidia
