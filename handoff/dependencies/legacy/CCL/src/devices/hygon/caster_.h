#ifndef INFINI_CCL_DEVICES_HYGON_CASTER_H_
#define INFINI_CCL_DEVICES_HYGON_CASTER_H_

#include "caster.h"
#include "data_type_.h"
#include "traits.h"

namespace infini::ccl {

// DTK exposes half/bfloat16 compound operators to `hipcc`, but they are not
// usable on host staging buffers in the MPI backend. Force these types through
// the bridge.
template <typename S, typename Op>
struct SupportsOp<half, S, Op, void> : std::false_type {};

template <typename S, typename Op>
struct SupportsOp<hip_bfloat16, S, Op, void> : std::false_type {};

template <>
struct HardwareCastImpl<Device::Type::kHygon, float, half> {
  __host__ __device__ static float Apply(half x) { return __half2float(x); }
};

template <>
struct HardwareCastImpl<Device::Type::kHygon, half, float> {
  __host__ __device__ static half Apply(float x) { return __float2half(x); }
};

template <>
struct HardwareCastImpl<Device::Type::kHygon, float, hip_bfloat16> {
  __host__ __device__ static float Apply(hip_bfloat16 x) {
    return static_cast<float>(x);
  }
};

template <>
struct HardwareCastImpl<Device::Type::kHygon, hip_bfloat16, float> {
  __host__ __device__ static hip_bfloat16 Apply(float x) {
    return hip_bfloat16(x);
  }
};

template <>
struct HardwareCastImpl<Device::Type::kHygon, hip_bfloat16, int> {
  __host__ __device__ static hip_bfloat16 Apply(int x) {
    return hip_bfloat16(static_cast<float>(x));
  }
};

template <>
struct HardwareCastImpl<Device::Type::kHygon, half, int> {
  __host__ __device__ static half Apply(int x) {
    return __float2half(static_cast<float>(x));
  }
};

template <>
struct HardwareCastImpl<Device::Type::kHygon, hip_bfloat16, double> {
  __host__ __device__ static hip_bfloat16 Apply(double x) {
    return hip_bfloat16(static_cast<float>(x));
  }
};

template <>
struct HardwareCastImpl<Device::Type::kHygon, half, double> {
  __host__ __device__ static half Apply(double x) {
    return __float2half(static_cast<float>(x));
  }
};

template <>
struct HardwareCastImpl<Device::Type::kHygon, half, hip_bfloat16> {
  __host__ __device__ static half Apply(hip_bfloat16 x) {
    return __float2half(static_cast<float>(x));
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_DEVICES_HYGON_CASTER_H_
