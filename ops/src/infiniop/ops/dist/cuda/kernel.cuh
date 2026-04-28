#include "../../../reduce/cuda/reduce.cuh"
#include <cmath>
#include <type_traits>

struct DistIndexing {
    static constexpr int kMaxNdim = 8;

    int ndim;
    int64_t shape[kMaxNdim];
    int64_t x1_strides[kMaxNdim];
    int64_t x2_strides[kMaxNdim];
};

namespace op::cuda {

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
} // namespace op::cuda
