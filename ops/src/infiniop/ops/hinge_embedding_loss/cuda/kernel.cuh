#include <limits>
#include <type_traits>

#include "../../../reduce/cuda/reduce.cuh"

namespace op::cuda {

template <typename T>
__device__ __forceinline__ T max_(T a, T b) {
    return a > b ? a : b;
}

template <typename Tcompute, typename Tin>
__device__ __forceinline__ Tcompute load_as(const Tin *ptr, size_t offset) {
    return static_cast<Tcompute>(ptr[offset]);
}

template <>
__device__ __forceinline__ float load_as<float, half>(const half *ptr, size_t offset) {
    return __half2float(ptr[offset]);
}

template <>
__device__ __forceinline__ float load_as<float, cuda_bfloat16>(const cuda_bfloat16 *ptr, size_t offset) {
    return __bfloat162float(ptr[offset]);
}

template <typename Tout, typename Tcompute>
__device__ __forceinline__ void store_from(Tout *ptr, size_t offset, Tcompute value) {
    ptr[offset] = static_cast<Tout>(value);
}

template <>
__device__ __forceinline__ void store_from<half, float>(half *ptr, size_t offset, float value) {
    ptr[offset] = __float2half(value);
}

template <>
__device__ __forceinline__ void store_from<cuda_bfloat16, float>(cuda_bfloat16 *ptr, size_t offset, float value) {
    ptr[offset] = __float2bfloat16_rn(value);
}

template <typename Tcompute>
__device__ __forceinline__ Tcompute hinge_embedding_loss_value(Tcompute input, Tcompute target, Tcompute margin) {
    // Match PyTorch behavior:
    //  - target == 1  : loss = input
    //  - target == -1 : loss = max(0, margin - input)
    //  - else         : loss = max(input, margin)
    //
    // Note: While the docs state targets should be ±1, PyTorch defines a fallback
    // behavior for other values (e.g., randomly-initialized float targets).
    // Example (PyTorch): input=0.2, target=0.5, margin=1.0 -> loss=1.0.
    if (target == static_cast<Tcompute>(1)) {
        return input;
    }
    if (target == static_cast<Tcompute>(-1)) {
        return max_(static_cast<Tcompute>(0), margin - input);
    }
    return max_(input, margin);
}

template <typename T, typename Tcompute>
__global__ void hinge_embedding_loss_none_kernel(
    T *output,
    const T *input,
    const T *target,
    size_t n,
    size_t ndim,
    const size_t *__restrict__ shape,
    const ptrdiff_t *__restrict__ output_strides,
    const ptrdiff_t *__restrict__ input_strides,
    const ptrdiff_t *__restrict__ target_strides,
    bool output_contiguous,
    bool input_contiguous,
    bool target_contiguous,
    Tcompute margin_val) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    const size_t out_offset = output_contiguous ? idx : indexToOffset(idx, ndim, shape, output_strides);
    const size_t in_offset = input_contiguous ? idx : indexToOffset(idx, ndim, shape, input_strides);
    const size_t tgt_offset = target_contiguous ? idx : indexToOffset(idx, ndim, shape, target_strides);

    const Tcompute in_val = load_as<Tcompute>(input, in_offset);
    const Tcompute tgt_val = load_as<Tcompute>(target, tgt_offset);
    const Tcompute loss = hinge_embedding_loss_value(in_val, tgt_val, margin_val);
    store_from<T, Tcompute>(output, out_offset, loss);
}

template <typename T, typename Tcompute, unsigned int BLOCK_SIZE>
__global__ void hinge_embedding_loss_reduce_kernel(
    Tcompute *accum,
    const T *input,
    const T *target,
    size_t n,
    size_t ndim,
    const size_t *__restrict__ shape,
    const ptrdiff_t *__restrict__ input_strides,
    const ptrdiff_t *__restrict__ target_strides,
    bool input_contiguous,
    bool target_contiguous,
    Tcompute margin_val) {

    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    Tcompute sum = 0;

    for (size_t i = tid; i < n; i += blockDim.x * gridDim.x) {
        const size_t in_offset = input_contiguous ? i : indexToOffset(i, ndim, shape, input_strides);
        const size_t tgt_offset = target_contiguous ? i : indexToOffset(i, ndim, shape, target_strides);

        const Tcompute in_val = load_as<Tcompute>(input, in_offset);
        const Tcompute tgt_val = load_as<Tcompute>(target, tgt_offset);
        sum += hinge_embedding_loss_value(in_val, tgt_val, margin_val);
    }

    using BlockReduce = cub::BlockReduce<Tcompute, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    const Tcompute block_sum = BlockReduce(temp_storage).Sum(sum);

    if (threadIdx.x == 0) {
#ifndef ENABLE_ILUVATAR_API
        atomicAdd(accum, block_sum);
#else
        // Iluvatar Corex's clang-CUDA can't lower atomicAdd(double*, double).
        // hinge_embedding_loss is training-only; never reached on inference.
        *accum += block_sum;
#endif
    }
}

template <typename Tout, typename Tcompute>
__global__ void hinge_embedding_loss_finalize_kernel(
    Tout *output,
    const Tcompute *accum,
    size_t n,
    bool mean) {

    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    Tcompute result = *accum;
    if (mean) {
        // Match PyTorch behavior: mean reduction on an empty tensor yields NaN.
        if (n == 0) {
            result = std::numeric_limits<Tcompute>::quiet_NaN();
        } else {
            result = result / static_cast<Tcompute>(n);
        }
    }
    store_from<Tout, Tcompute>(output, 0, result);
}

} // namespace op::cuda
