#ifndef INFINI_OPS_THEAD_CASTER__H_
#define INFINI_OPS_THEAD_CASTER__H_

#include "native/cuda/caster.cuh"
#include "native/cuda/thead/data_type_.h"

namespace infini::ops {
namespace detail {

template <>
struct ToFloat<Device::Type::kThead, half> {
  __host__ __device__ float operator()(half x) { return __half2float(x); }
};

template <>
struct ToFloat<Device::Type::kThead, __nv_bfloat16> {
  __host__ __device__ float operator()(__nv_bfloat16 x) {
    return __bfloat162float(x);
  }
};

template <>
struct FromFloat<Device::Type::kThead, half> {
  __host__ __device__ half operator()(float f) { return __float2half(f); }
};

template <>
struct FromFloat<Device::Type::kThead, __nv_bfloat16> {
  __host__ __device__ __nv_bfloat16 operator()(float f) {
    return __float2bfloat16(f);
  }
};

template <>
struct HardwareCast<Device::Type::kThead, __nv_bfloat16, int> {
  inline static constexpr bool kSupported = true;
  __host__ __device__ __nv_bfloat16 operator()(int x) {
    return __int2bfloat16_rn(x);
  }
};

template <>
struct HardwareCast<Device::Type::kThead, half, int> {
  inline static constexpr bool kSupported = true;
  __host__ __device__ half operator()(int x) { return __int2half_rn(x); }
};

template <>
struct HardwareCast<Device::Type::kThead, __nv_bfloat16, double> {
  inline static constexpr bool kSupported = true;
  __host__ __device__ __nv_bfloat16 operator()(double x) {
    return __double2bfloat16(x);
  }
};

template <>
struct HardwareCast<Device::Type::kThead, half, double> {
  inline static constexpr bool kSupported = true;
  __host__ __device__ half operator()(double x) { return __double2half(x); }
};

template <>
struct HardwareCast<Device::Type::kThead, half, __nv_bfloat16> {
  inline static constexpr bool kSupported = true;
  __host__ __device__ half operator()(__nv_bfloat16 x) { return __half(x); }
};

}  // namespace detail

template <>
struct Caster<Device::Type::kThead> : CudaCasterImpl<Device::Type::kThead> {};

}  // namespace infini::ops

#endif
