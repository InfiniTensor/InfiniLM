#ifndef INFINI_OPS_HYGON_CASTER_CUH_
#define INFINI_OPS_HYGON_CASTER_CUH_

#include <infini/rt/hygon/data_type_.h>

#include "native/cuda/caster.cuh"

namespace infini::ops {

namespace detail {

template <>
struct ToFloat<Device::Type::kHygon, half> {
  __host__ __device__ float operator()(half x) { return __half2float(x); }
};

template <>
struct ToFloat<Device::Type::kHygon, __nv_bfloat16> {
  __host__ __device__ float operator()(__nv_bfloat16 x) {
    return __bfloat162float(x);
  }
};

template <>
struct FromFloat<Device::Type::kHygon, half> {
  __host__ __device__ half operator()(float f) { return __float2half(f); }
};

template <>
struct FromFloat<Device::Type::kHygon, __nv_bfloat16> {
  __host__ __device__ __nv_bfloat16 operator()(float f) {
    return __float2bfloat16(f);
  }
};

}  // namespace detail

template <>
struct Caster<Device::Type::kHygon> : CudaCasterImpl<Device::Type::kHygon> {};

}  // namespace infini::ops

#endif
