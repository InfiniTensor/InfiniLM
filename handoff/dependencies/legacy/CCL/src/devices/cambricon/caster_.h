#ifndef INFINI_CCL_DEVICES_CAMBRICON_CASTER_H_
#define INFINI_CCL_DEVICES_CAMBRICON_CASTER_H_

#include "caster.h"
#include "data_type_.h"

namespace infini::ccl {

template <>
struct HardwareCastImpl<Device::Type::kCambricon, float, CambriconFP16> {
  static float Apply(CambriconFP16 x);
};

template <>
struct HardwareCastImpl<Device::Type::kCambricon, CambriconFP16, float> {
  static CambriconFP16 Apply(float x);
};

template <>
struct HardwareCastImpl<Device::Type::kCambricon, CambriconFP16, int> {
  static CambriconFP16 Apply(int x);
};

template <>
struct HardwareCastImpl<Device::Type::kCambricon, float, CambriconBF16> {
  static float Apply(CambriconBF16 x);
};

template <>
struct HardwareCastImpl<Device::Type::kCambricon, CambriconBF16, float> {
  static CambriconBF16 Apply(float x);
};

template <>
struct HardwareCastImpl<Device::Type::kCambricon, CambriconBF16, int> {
  static CambriconBF16 Apply(int x);
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_DEVICES_CAMBRICON_CASTER_H_
