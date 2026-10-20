#ifndef INFINI_OPS_MOORE_CASTER__H_
#define INFINI_OPS_MOORE_CASTER__H_

#include "native/cuda/caster.cuh"
#include "native/cuda/moore/data_type_.h"

namespace infini::ops {

namespace detail {

template <>
struct ToFloat<Device::Type::kMoore, half> {
  __host__ __device__ float operator()(half x) { return __half2float(x); }
};

template <>
struct ToFloat<Device::Type::kMoore, __mt_bfloat16> {
  __host__ __device__ float operator()(__mt_bfloat16 x) {
    return __bfloat162float(x);
  }
};

template <>
struct FromFloat<Device::Type::kMoore, half> {
  __host__ __device__ half operator()(float f) { return __float2half_rn(f); }
};

template <>
struct FromFloat<Device::Type::kMoore, __mt_bfloat16> {
  __host__ __device__ __mt_bfloat16 operator()(float f) {
    return __float2bfloat16_rn(f);
  }
};

template <>
struct HardwareCast<Device::Type::kMoore, half, int> {
  inline static constexpr bool kSupported = true;
  __host__ __device__ half operator()(int x) { return __int2half_rn(x); }
};

template <>
struct HardwareCast<Device::Type::kMoore, __mt_bfloat16, double> {
  inline static constexpr bool kSupported = true;
  __host__ __device__ __mt_bfloat16 operator()(double x) {
    return __double2bfloat16(x);
  }
};

template <>
struct HardwareCast<Device::Type::kMoore, half, double> {
  inline static constexpr bool kSupported = true;
  __host__ __device__ half operator()(double x) { return __double2half(x); }
};

template <>
struct HardwareCast<Device::Type::kMoore, half, __mt_bfloat16> {
  inline static constexpr bool kSupported = true;
  __host__ __device__ half operator()(__mt_bfloat16 x) {
    return __float2half_rn(__bfloat162float(x));
  }
};

}  // namespace detail

template <>
struct Caster<Device::Type::kMoore> : CudaCasterImpl<Device::Type::kMoore> {};

}  // namespace infini::ops

#endif
