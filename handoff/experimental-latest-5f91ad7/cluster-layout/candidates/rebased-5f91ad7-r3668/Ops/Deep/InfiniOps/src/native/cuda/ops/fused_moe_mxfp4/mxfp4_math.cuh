#ifndef INFINIOPS_MXFP4_MATH_CUH_
#define INFINIOPS_MXFP4_MATH_CUH_

#include <cmath>
#include <cstddef>
#include <cstdint>

__device__ __forceinline__ float mxfp4DecodeE2M1(uint8_t code) {
    constexpr float magnitudes[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float magnitude = magnitudes[code & 0x7];
    return code & 0x8 ? -magnitude : magnitude;
}

__device__ __forceinline__ void mxfp4DecodePair(uint8_t packed,
                                                uint8_t scale,
                                                float &low,
                                                float &high) {
    const int exponent = static_cast<int>(scale) - 127;
    low = ldexpf(mxfp4DecodeE2M1(packed & 0xf), exponent);
    high = ldexpf(mxfp4DecodeE2M1(packed >> 4), exponent);
}

template <typename T>
__device__ __forceinline__ float mxfp4Load(const T *ptr, size_t index) {
    return static_cast<float>(ptr[index]);
}

template <>
__device__ __forceinline__ float mxfp4Load<half>(const half *ptr, size_t index) {
    return __half2float(ptr[index]);
}

template <>
__device__ __forceinline__ float mxfp4Load<__nv_bfloat16>(const __nv_bfloat16 *ptr,
                                                          size_t index) {
    return __bfloat162float(ptr[index]);
}

template <typename T>
__device__ __forceinline__ T mxfp4Store(float value) {
    return static_cast<T>(value);
}

template <>
__device__ __forceinline__ half mxfp4Store<half>(float value) {
    return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 mxfp4Store<__nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}

template <size_t N>
__device__ __forceinline__ void mxfp4BlockReduce(float (&values)[N], float *scratch) {
    for (size_t i = 0; i < N; ++i) {
        scratch[i * blockDim.x + threadIdx.x] = values[i];
    }
    __syncthreads();

    for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            for (size_t i = 0; i < N; ++i) {
                scratch[i * blockDim.x + threadIdx.x]
                    += scratch[i * blockDim.x + threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    for (size_t i = 0; i < N; ++i) {
        values[i] = scratch[i * blockDim.x];
    }
    __syncthreads();
}

#endif
