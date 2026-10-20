#ifndef INFINI_OPS_MARS_CASTER__H_
#define INFINI_OPS_MARS_CASTER__H_

#include "native/cuda/caster.cuh"
#include "native/cuda/mars/data_type_.h"

namespace infini::ops {

namespace detail {

template <>
struct ToFloat<Device::Type::kMars, __half> {
  __host__ __device__ float operator()(__half x) { return __half2float(x); }
};

template <>
struct ToFloat<Device::Type::kMars, __hpcc_bfloat16> {
  __host__ __device__ float operator()(__hpcc_bfloat16 x) {
    return __bfloat162float(x);
  }
};

template <>
struct FromFloat<Device::Type::kMars, __half> {
  __host__ __device__ __half operator()(float f) { return __float2half(f); }
};

template <>
struct FromFloat<Device::Type::kMars, __hpcc_bfloat16> {
  __host__ __device__ __hpcc_bfloat16 operator()(float f) {
    return __float2bfloat16(f);
  }
};

template <>
struct HardwareCast<Device::Type::kMars, __hpcc_bfloat16, int> {
  inline static constexpr bool kSupported = true;
  __host__ __device__ __hpcc_bfloat16 operator()(int x) {
    return __int2bfloat16_rn(x);
  }
};

template <>
struct HardwareCast<Device::Type::kMars, __half, int> {
  inline static constexpr bool kSupported = true;
  __host__ __device__ __half operator()(int x) { return __int2half_rn(x); }
};

template <>
struct HardwareCast<Device::Type::kMars, __hpcc_bfloat16, double> {
  inline static constexpr bool kSupported = true;
  __host__ __device__ __hpcc_bfloat16 operator()(double x) {
    return __double2bfloat16(x);
  }
};

template <>
struct HardwareCast<Device::Type::kMars, __half, double> {
  inline static constexpr bool kSupported = true;
  __host__ __device__ __half operator()(double x) { return __double2half(x); }
};

template <>
struct HardwareCast<Device::Type::kMars, __half, __hpcc_bfloat16> {
  inline static constexpr bool kSupported = true;
  __host__ __device__ __half operator()(__hpcc_bfloat16 x) {
    return __float2half_rn(__bfloat162float(x));
  }
};

}  // namespace detail

template <>
struct Caster<Device::Type::kMars> : CudaCasterImpl<Device::Type::kMars> {};

}  // namespace infini::ops

#endif
