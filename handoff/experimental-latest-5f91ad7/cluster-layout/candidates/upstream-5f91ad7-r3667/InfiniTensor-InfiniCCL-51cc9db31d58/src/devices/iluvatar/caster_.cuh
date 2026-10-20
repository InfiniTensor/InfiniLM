#ifndef INFINI_CCL_DEVICES_ILUVATAR_CASTER_CUH_
#define INFINI_CCL_DEVICES_ILUVATAR_CASTER_CUH_

#include "caster.h"
#include "data_type_.h"
#include "traits.h"

namespace infini::ccl {

// Explicitly tell the trait system that host-side math on these types is
// unworkable on Iluvatar.
template <typename S, typename Op>
struct SupportsOp<half, S, Op, void> : std::false_type {};

template <typename S, typename Op>
struct SupportsOp<__nv_bfloat16, S, Op, void> : std::false_type {};

template <>
struct HardwareCastImpl<Device::Type::kIluvatar, float, half> {
  __host__ __device__ static float Apply(half x) { return __half2float(x); }
};

template <>
struct HardwareCastImpl<Device::Type::kIluvatar, float, __nv_bfloat16> {
  __host__ __device__ static float Apply(__nv_bfloat16 x) {
    return __bfloat162float(x);
  }
};

template <>
struct HardwareCastImpl<Device::Type::kIluvatar, half, float> {
  __host__ __device__ static half Apply(float x) { return __float2half(x); }
};

template <>
struct HardwareCastImpl<Device::Type::kIluvatar, __nv_bfloat16, float> {
  __host__ __device__ static __nv_bfloat16 Apply(float x) {
    return __float2bfloat16(x);
  }
};

template <>
struct HardwareCastImpl<Device::Type::kIluvatar, __nv_bfloat16, double> {
  __host__ __device__ static __nv_bfloat16 Apply(double x) {
    return __double2bfloat16(x);
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_DEVICES_ILUVATAR_CASTER_CUH_
