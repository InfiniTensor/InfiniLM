#ifndef __ACOS_CUDA_H__
#define __ACOS_CUDA_H__

#include <cmath>
#include <math.h>
#include <type_traits>

namespace op::acos::cuda {

// ----------------------
// Fast acos approximation
// ----------------------
__device__ __forceinline__ float fast_acosf(float x) {
    // 高性能多项式近似 acos(x)
    float ax = fabsf(x);
    float t = sqrtf(1.0f - ax);
    float r = ((-0.0187293f * ax + 0.0742610f) * ax - 0.2121144f) * ax + 1.5707288f;
    return (x >= 0.0f ? t * r : 3.14159265358979323846f - t * r);
}

// ----------------------
// float kernel (F32)
// ----------------------
template <typename T>
__device__ __forceinline__ T acos_impl(T val);

template <>
__device__ __forceinline__ float acos_impl<float>(float val) {
    return fast_acosf(val);
}

// ----------------------
// half kernel (F16)
// ----------------------
template <>
__device__ __forceinline__ half acos_impl<half>(half val) {
#if (__CUDA_ARCH__ >= 530)
    float f = __half2float(val);
    return __float2half(fast_acosf(f));
#else
    float f = __half2float(val);
    return __float2half(fast_acosf(f));
#endif
}

// ----------------------
// half2 kernel (F16x2 vectorized)
// ----------------------
template <>
__device__ __forceinline__ half2 acos_impl<half2>(half2 val) {
#if (__CUDA_ARCH__ >= 530)
    float2 f = __half22float2(val);
    f.x = fast_acosf(f.x);
    f.y = fast_acosf(f.y);
    return __float22half2_rn(f);
#else
    float2 f = __half22float2(val);
    f.x = fast_acosf(f.x);
    f.y = fast_acosf(f.y);
    return __float22half2_rn(f);
#endif
}

// ----------------------
// bfloat16 kernel (BF16)
// ----------------------
template <>
__device__ __forceinline__ cuda_bfloat16 acos_impl<cuda_bfloat16>(cuda_bfloat16 val) {
    float f = __bfloat162float(val);
    return __float2bfloat16(fast_acosf(f));
}

// ----------------------
// Fallback kernel
// ----------------------
template <typename T>
__device__ __forceinline__ T acos_impl(T val) {
    return static_cast<T>(fast_acosf(static_cast<float>(val)));
}

// ----------------------
// AcosOp struct
// ----------------------
struct AcosOp {
    static constexpr size_t num_inputs = 1;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &a) const {
        return acos_impl(a);
    }
};

} // namespace op::acos::cuda

#endif // __ACOS_CUDA_H__
