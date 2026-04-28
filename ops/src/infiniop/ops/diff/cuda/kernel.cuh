#include <type_traits>

struct Diff1Indexing {
    static constexpr int kMaxNdim = 8;

    int ndim;
    int dim;
    int64_t out_shape[kMaxNdim];
    int64_t in_strides[kMaxNdim];
    int64_t out_strides[kMaxNdim];
};

namespace op::cuda {

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
} // namespace op::cuda
