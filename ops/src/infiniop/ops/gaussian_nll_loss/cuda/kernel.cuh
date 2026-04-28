#pragma once
#include "../../../reduce/cuda/reduce.cuh"
#include <cmath>
#include <type_traits>

namespace op::cuda {

constexpr int kGaussianNllMaxDims = 8;

struct GaussianNllTensorMeta {
    int ndim;
    size_t shape[kGaussianNllMaxDims];
    ptrdiff_t strides[kGaussianNllMaxDims]; // strides in elements
};

template <typename T>
struct GaussianNllTypeTag {};

template <typename Tcompute>
__device__ __forceinline__ Tcompute gaussian_nll_to_compute(const half v) {
    return static_cast<Tcompute>(__half2float(v));
}

template <typename Tcompute>
__device__ __forceinline__ Tcompute gaussian_nll_to_compute(const cuda_bfloat16 v) {
    return static_cast<Tcompute>(__bfloat162float(v));
}

template <typename Tcompute, typename T>
__device__ __forceinline__ Tcompute gaussian_nll_to_compute(const T v) {
    return static_cast<Tcompute>(v);
}

__device__ __forceinline__ half gaussian_nll_from_compute(const float v, GaussianNllTypeTag<half>) {
    return __float2half_rn(v);
}

__device__ __forceinline__ cuda_bfloat16 gaussian_nll_from_compute(const float v, GaussianNllTypeTag<cuda_bfloat16>) {
    return __float2bfloat16_rn(v);
}

template <typename Tcompute, typename T>
__device__ __forceinline__ T gaussian_nll_from_compute(const Tcompute v, GaussianNllTypeTag<T>) {
    return static_cast<T>(v);
}

__device__ __forceinline__ size_t gaussian_nll_offset(size_t flat, const GaussianNllTensorMeta &meta) {
    size_t res = 0;
    for (size_t i = meta.ndim; i-- > 0;) {
        res += (flat % meta.shape[i]) * meta.strides[i];
        flat /= meta.shape[i];
    }
    return res;
}

template <typename T, typename Tcompute>
__global__ void gaussian_nll_loss_kernel(
    T *output,
    const T *input,
    const T *target,
    const T *var,
    size_t n,
    GaussianNllTensorMeta out_meta,
    GaussianNllTensorMeta in_meta,
    GaussianNllTensorMeta tgt_meta,
    GaussianNllTensorMeta var_meta,
    Tcompute eps_val,
    int full) {

    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    const size_t out_off = gaussian_nll_offset(idx, out_meta);
    const size_t in_off = gaussian_nll_offset(idx, in_meta);
    const size_t tgt_off = gaussian_nll_offset(idx, tgt_meta);
    const size_t var_off = gaussian_nll_offset(idx, var_meta);

    const Tcompute diff = gaussian_nll_to_compute<Tcompute>(input[in_off]) - gaussian_nll_to_compute<Tcompute>(target[tgt_off]);
    Tcompute var_val = gaussian_nll_to_compute<Tcompute>(var[var_off]);
    if (var_val < eps_val) {
        var_val = eps_val;
    }
    Tcompute loss = Tcompute(0.5) * (log(var_val) + (diff * diff) / var_val);
    if (full) {
        loss += Tcompute(0.9189385332046727); // log(2*pi)/2
    }
    output[out_off] = gaussian_nll_from_compute(loss, GaussianNllTypeTag<T>{});
}

template <typename T, typename Tcompute>
__global__ void gaussian_nll_loss_reduce_kernel(
    Tcompute *output,
    const T *input,
    const T *target,
    const T *var,
    size_t n,
    GaussianNllTensorMeta in_meta,
    GaussianNllTensorMeta tgt_meta,
    GaussianNllTensorMeta var_meta,
    Tcompute eps_val,
    int full) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    Tcompute sum = 0;

    Tcompute log_2pi = full ? Tcompute(0.9189385332046727) : Tcompute(0.0);

    for (size_t i = idx; i < n; i += blockDim.x * gridDim.x) {
        const size_t in_off = gaussian_nll_offset(i, in_meta);
        const size_t tgt_off = gaussian_nll_offset(i, tgt_meta);
        const size_t var_off = gaussian_nll_offset(i, var_meta);

        const Tcompute diff = gaussian_nll_to_compute<Tcompute>(input[in_off]) - gaussian_nll_to_compute<Tcompute>(target[tgt_off]);
        Tcompute var_val = gaussian_nll_to_compute<Tcompute>(var[var_off]);
        if (var_val < eps_val) {
            var_val = eps_val;
        }
        const Tcompute loss = Tcompute(0.5) * (log(var_val) + (diff * diff) / var_val) + log_2pi;
        sum += loss;
    }

    using BlockReduce = cub::BlockReduce<Tcompute, 256>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    Tcompute block_sum = BlockReduce(temp_storage).Sum(sum);

    if (threadIdx.x == 0) {
#ifndef ENABLE_ILUVATAR_API
        atomicAdd(output, block_sum);
#else
        // Iluvatar Corex's clang-CUDA can't lower atomicAdd(double*, double).
        // gaussian_nll_loss is a training-side op; never reached on inference.
        *output += block_sum;
#endif
    }
}

template <typename Tout, typename Tcompute>
__global__ void gaussian_nll_loss_finalize_kernel(
    Tout *output,
    const Tcompute *accum,
    Tcompute scale) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const Tcompute v = (*accum) * scale;
        output[0] = gaussian_nll_from_compute(v, GaussianNllTypeTag<Tout>{});
    }
}

// ---------------------------------------------------------------------------
// Compatibility wrappers for backends that still reference the older contiguous
// kernel signatures.
// ---------------------------------------------------------------------------

template <typename T>
__global__ void gaussian_nll_loss_kernel(
    T *output,
    const T *input,
    const T *target,
    const T *var,
    size_t n,
    T eps_val,
    int full) {

    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    using Tcompute = std::conditional_t<std::is_same_v<T, double>, double, float>;
    const Tcompute eps_c = gaussian_nll_to_compute<Tcompute>(eps_val);
    const Tcompute diff = gaussian_nll_to_compute<Tcompute>(input[idx]) - gaussian_nll_to_compute<Tcompute>(target[idx]);
    Tcompute var_val = gaussian_nll_to_compute<Tcompute>(var[idx]);
    if (var_val < eps_c) {
        var_val = eps_c;
    }
    Tcompute loss = Tcompute(0.5) * (log(var_val) + (diff * diff) / var_val);
    if (full) {
        loss += Tcompute(0.9189385332046727);
    }
    output[idx] = gaussian_nll_from_compute(loss, GaussianNllTypeTag<T>{});
}

template <typename T, typename Tcompute>
__global__ void gaussian_nll_loss_reduce_kernel(
    T *output,
    const T *input,
    const T *target,
    const T *var,
    size_t n,
    Tcompute eps_val,
    int full) {

    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    Tcompute sum = 0;

    const Tcompute log_2pi = full ? Tcompute(0.9189385332046727) : Tcompute(0.0);
    for (size_t i = idx; i < n; i += blockDim.x * gridDim.x) {
        const Tcompute diff = gaussian_nll_to_compute<Tcompute>(input[i]) - gaussian_nll_to_compute<Tcompute>(target[i]);
        Tcompute var_val = gaussian_nll_to_compute<Tcompute>(var[i]);
        if (var_val < eps_val) {
            var_val = eps_val;
        }
        const Tcompute loss = Tcompute(0.5) * (log(var_val) + (diff * diff) / var_val) + log_2pi;
        sum += loss;
    }

    using BlockReduce = cub::BlockReduce<Tcompute, 256>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    const Tcompute block_sum = BlockReduce(temp_storage).Sum(sum);

    if (threadIdx.x == 0) {
#ifndef ENABLE_ILUVATAR_API
        atomicAdd(reinterpret_cast<Tcompute *>(output), block_sum);
#else
        *reinterpret_cast<Tcompute *>(output) += block_sum;
#endif
    }
}

} // namespace op::cuda
