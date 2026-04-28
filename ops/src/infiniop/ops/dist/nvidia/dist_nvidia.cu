#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../tensor.h"
#include "dist_nvidia.cuh"
#include <cmath>
#include <cstdint>
#include <cub/block/block_reduce.cuh>
#include <type_traits>

namespace op::dist::nvidia {

Descriptor::~Descriptor() = default;

struct DistIndexing {
    static constexpr int kMaxNdim = 8;

    int ndim;
    int64_t shape[kMaxNdim];
    int64_t x1_strides[kMaxNdim];
    int64_t x2_strides[kMaxNdim];
};

template <typename T>
__device__ __forceinline__ float to_f32(T v) {
    return static_cast<float>(v);
}

template <>
__device__ __forceinline__ float to_f32<half>(half v) {
    return __half2float(v);
}

template <>
__device__ __forceinline__ float to_f32<cuda_bfloat16>(cuda_bfloat16 v) {
    return __bfloat162float(v);
}

template <typename Tdata, typename Tcompute>
__device__ __forceinline__ Tdata cast_out(Tcompute v) {
    return static_cast<Tdata>(v);
}

template <>
__device__ __forceinline__ half cast_out<half, float>(float v) {
    return __float2half(v);
}

template <>
__device__ __forceinline__ cuda_bfloat16 cast_out<cuda_bfloat16, float>(float v) {
    return __float2bfloat16_rn(v);
}

template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
__global__ void dist_strided_kernel(
    Tcompute *result,
    const Tdata *x1,
    const Tdata *x2,
    size_t n,
    double p,
    DistIndexing indexing) {

    Tcompute thread_val = static_cast<Tcompute>(0);

    for (size_t linear = static_cast<size_t>(threadIdx.x); linear < n; linear += BLOCK_SIZE) {
        int64_t idx[DistIndexing::kMaxNdim] = {0};
        size_t tmp = linear;
        for (int d = indexing.ndim - 1; d >= 0; --d) {
            const int64_t s = indexing.shape[d];
            idx[d] = static_cast<int64_t>(tmp % static_cast<size_t>(s));
            tmp /= static_cast<size_t>(s);
        }

        int64_t off1 = 0;
        int64_t off2 = 0;
        for (int d = 0; d < indexing.ndim; ++d) {
            off1 += idx[d] * indexing.x1_strides[d];
            off2 += idx[d] * indexing.x2_strides[d];
        }

        Tcompute diff;
        if constexpr (std::is_same_v<Tcompute, double>) {
            diff = static_cast<double>(x1[off1]) - static_cast<double>(x2[off2]);
        } else {
            diff = static_cast<Tcompute>(to_f32(x1[off1]) - to_f32(x2[off2]));
        }
        const Tcompute abs_diff = fabs(diff);

        if (p == 0.0) {
            if (abs_diff > static_cast<Tcompute>(1e-10)) {
                thread_val += static_cast<Tcompute>(1);
            }
        } else if (isinf(p)) {
            thread_val = fmax(thread_val, abs_diff);
        } else {
            thread_val += pow(abs_diff, static_cast<Tcompute>(p));
        }
    }

    using BlockReduce = cub::BlockReduce<Tcompute, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    if (isinf(p)) {
        struct MaxOp {
            __device__ __forceinline__ Tcompute operator()(Tcompute a, Tcompute b) const {
                return a > b ? a : b;
            }
        };
        const Tcompute block_max = BlockReduce(temp_storage).Reduce(thread_val, MaxOp{});
        if (threadIdx.x == 0) {
            *result = block_max;
        }
        return;
    }

    const Tcompute block_sum = BlockReduce(temp_storage).Sum(thread_val);
    if (threadIdx.x == 0) {
        if (p == 0.0) {
            *result = block_sum;
        } else {
            *result = pow(block_sum, static_cast<Tcompute>(1.0 / p));
        }
    }
}

template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
__global__ void dist_strided_out_kernel(
    Tdata *out,
    const Tdata *x1,
    const Tdata *x2,
    size_t n,
    double p,
    DistIndexing indexing) {

    Tcompute thread_val = static_cast<Tcompute>(0);

    for (size_t linear = static_cast<size_t>(threadIdx.x); linear < n; linear += BLOCK_SIZE) {
        int64_t idx[DistIndexing::kMaxNdim] = {0};
        size_t tmp = linear;
        for (int d = indexing.ndim - 1; d >= 0; --d) {
            const int64_t s = indexing.shape[d];
            idx[d] = static_cast<int64_t>(tmp % static_cast<size_t>(s));
            tmp /= static_cast<size_t>(s);
        }

        int64_t off1 = 0;
        int64_t off2 = 0;
        for (int d = 0; d < indexing.ndim; ++d) {
            off1 += idx[d] * indexing.x1_strides[d];
            off2 += idx[d] * indexing.x2_strides[d];
        }

        Tcompute diff;
        if constexpr (std::is_same_v<Tcompute, double>) {
            diff = static_cast<double>(x1[off1]) - static_cast<double>(x2[off2]);
        } else {
            diff = static_cast<Tcompute>(to_f32(x1[off1]) - to_f32(x2[off2]));
        }
        const Tcompute abs_diff = fabs(diff);

        if (p == 0.0) {
            if (abs_diff > static_cast<Tcompute>(1e-10)) {
                thread_val += static_cast<Tcompute>(1);
            }
        } else if (isinf(p)) {
            thread_val = fmax(thread_val, abs_diff);
        } else {
            thread_val += pow(abs_diff, static_cast<Tcompute>(p));
        }
    }

    using BlockReduce = cub::BlockReduce<Tcompute, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    if (isinf(p)) {
        struct MaxOp {
            __device__ __forceinline__ Tcompute operator()(Tcompute a, Tcompute b) const {
                return a > b ? a : b;
            }
        };
        const Tcompute block_max = BlockReduce(temp_storage).Reduce(thread_val, MaxOp{});
        if (threadIdx.x == 0) {
            *out = cast_out<Tdata, Tcompute>(block_max);
        }
        return;
    }

    const Tcompute block_sum = BlockReduce(temp_storage).Sum(thread_val);
    if (threadIdx.x == 0) {
        if (p == 0.0) {
            *out = cast_out<Tdata, Tcompute>(block_sum);
        } else {
            *out = cast_out<Tdata, Tcompute>(pow(block_sum, static_cast<Tcompute>(1.0 / p)));
        }
    }
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x1_desc,
    infiniopTensorDescriptor_t x2_desc,
    double p) {

    auto dtype = x1_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16);

    auto x1_shape = x1_desc->shape();
    auto x2_shape = x2_desc->shape();
    auto y_shape = y_desc->shape();

    if (x1_shape != x2_shape) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    if (y_shape.size() != 0 && (y_shape.size() != 1 || y_shape[0] != 1)) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    const size_t ndim = x1_desc->ndim();
    if (ndim > static_cast<size_t>(DistIndexing::kMaxNdim)) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    size_t input_size = x1_desc->numel();
    *desc_ptr = new Descriptor(dtype, input_size, p, ndim, x1_shape, x1_desc->strides(), x2_desc->strides(),
                               handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x1,
    const void *x2,
    void *stream) const {

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    constexpr int BLOCK_SIZE = 256;

    DistIndexing indexing{};
    indexing.ndim = static_cast<int>(_ndim);
    for (int d = 0; d < DistIndexing::kMaxNdim; ++d) {
        indexing.shape[d] = 1;
        indexing.x1_strides[d] = 0;
        indexing.x2_strides[d] = 0;
    }
    for (size_t d = 0; d < _ndim; ++d) {
        indexing.shape[d] = static_cast<int64_t>(_shape[d]);
        indexing.x1_strides[d] = static_cast<int64_t>(_x1_strides[d]);
        indexing.x2_strides[d] = static_cast<int64_t>(_x2_strides[d]);
    }

    switch (_dtype) {
    case INFINI_DTYPE_F16: {
        dist_strided_out_kernel<BLOCK_SIZE, half, float><<<1, BLOCK_SIZE, 0, cuda_stream>>>(
            reinterpret_cast<half *>(y),
            reinterpret_cast<const half *>(x1), reinterpret_cast<const half *>(x2),
            _input_size, _p, indexing);
        break;
    }
    case INFINI_DTYPE_BF16: {
        dist_strided_out_kernel<BLOCK_SIZE, cuda_bfloat16, float><<<1, BLOCK_SIZE, 0, cuda_stream>>>(
            reinterpret_cast<cuda_bfloat16 *>(y),
            reinterpret_cast<const cuda_bfloat16 *>(x1), reinterpret_cast<const cuda_bfloat16 *>(x2),
            _input_size, _p, indexing);
        break;
    }
    case INFINI_DTYPE_F32: {
        float *result_f = reinterpret_cast<float *>(y);
        dist_strided_kernel<BLOCK_SIZE, float, float><<<1, BLOCK_SIZE, 0, cuda_stream>>>(
            result_f, reinterpret_cast<const float *>(x1), reinterpret_cast<const float *>(x2),
            _input_size, _p, indexing);
        break;
    }
    case INFINI_DTYPE_F64: {
        double *result_d = reinterpret_cast<double *>(y);
        dist_strided_kernel<BLOCK_SIZE, double, double><<<1, BLOCK_SIZE, 0, cuda_stream>>>(
            result_d, reinterpret_cast<const double *>(x1), reinterpret_cast<const double *>(x2),
            _input_size, _p, indexing);
        break;
    }
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::dist::nvidia
