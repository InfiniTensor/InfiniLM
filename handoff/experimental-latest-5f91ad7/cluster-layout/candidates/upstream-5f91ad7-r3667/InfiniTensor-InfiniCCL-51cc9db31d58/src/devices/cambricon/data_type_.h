#ifndef INFINI_CCL_DEVICES_CAMBRICON_DATA_TYPE_H_
#define INFINI_CCL_DEVICES_CAMBRICON_DATA_TYPE_H_

#include <cstdint>

#include "data_type_impl.h"

namespace infini::ccl {

// Unique tag types to force completely distinct template signatures.
struct CambriconFP16 {
  uint16_t bits;
};
struct CambriconBF16 {
  uint16_t bits;
};

template <>
struct TypeMap<Device::Type::kCambricon, DataType::kFloat16> {
  using type = CambriconFP16;
};

template <>
struct TypeMap<Device::Type::kCambricon, DataType::kBFloat16> {
  using type = CambriconBF16;
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_DEVICES_CAMBRICON_DATA_TYPE_H_
