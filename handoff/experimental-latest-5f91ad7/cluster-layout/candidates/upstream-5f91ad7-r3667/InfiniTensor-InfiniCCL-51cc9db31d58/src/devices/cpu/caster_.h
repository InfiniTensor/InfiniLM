#ifndef INFINI_CCL_DEVICES_CPU_CASTER_H_
#define INFINI_CCL_DEVICES_CPU_CASTER_H_

#include "caster.h"
#include "data_type_.h"

namespace infini::ccl {

template <>
struct HardwareCastImpl<Device::Type::kCpu, float, Float16> {
  static float Apply(Float16 x) { return x.ToFloat(); }
};

template <>
struct HardwareCastImpl<Device::Type::kCpu, Float16, float> {
  static Float16 Apply(float x) { return Float16::FromFloat(x); }
};

template <>
struct HardwareCastImpl<Device::Type::kCpu, float, BFloat16> {
  static float Apply(BFloat16 x) { return x.ToFloat(); }
};

template <>
struct HardwareCastImpl<Device::Type::kCpu, BFloat16, float> {
  static BFloat16 Apply(float x) { return BFloat16::FromFloat(x); }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_DEVICES_CPU_CASTER_H_
