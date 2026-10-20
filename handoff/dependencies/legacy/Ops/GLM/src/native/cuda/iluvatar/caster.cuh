#ifndef INFINI_OPS_ILUVATAR_CASTER__H_
#define INFINI_OPS_ILUVATAR_CASTER__H_

#include "native/cuda/caster.cuh"
#include "native/cuda/iluvatar/data_type_.h"

namespace infini::ops {

namespace detail {

template <>
struct ToFloat<Device::Type::kIluvatar, half> {
  __host__ __device__ float operator()(half x) { return __half2float(x); }
};

template <>
struct ToFloat<Device::Type::kIluvatar, __nv_bfloat16> {
  __host__ __device__ float operator()(__nv_bfloat16 x) {
    return __bfloat162float(x);
  }
};

template <>
struct FromFloat<Device::Type::kIluvatar, half> {
  __host__ __device__ half operator()(float f) { return __float2half(f); }
};

template <>
struct FromFloat<Device::Type::kIluvatar, __nv_bfloat16> {
  __host__ __device__ __nv_bfloat16 operator()(float f) {
    return __float2bfloat16(f);
  }
};

template <>
struct HardwareCast<Device::Type::kIluvatar, __nv_bfloat16, double> {
  inline static constexpr bool kSupported = true;
  __host__ __device__ __nv_bfloat16 operator()(double x) {
    return __double2bfloat16(x);
  }
};

}  // namespace detail

template <>
struct Caster<Device::Type::kIluvatar>
    : CudaCasterImpl<Device::Type::kIluvatar> {};

}  // namespace infini::ops

#endif
