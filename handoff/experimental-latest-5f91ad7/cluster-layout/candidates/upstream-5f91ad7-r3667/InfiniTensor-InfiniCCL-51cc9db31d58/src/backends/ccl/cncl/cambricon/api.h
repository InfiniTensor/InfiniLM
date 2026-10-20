#ifndef INFINI_CCL_BACKENDS_CCL_CNCL_CAMBRICON_API_H_
#define INFINI_CCL_BACKENDS_CCL_CNCL_CAMBRICON_API_H_

#include "backends/ccl/cncl/api.h"
#include "devices/cambricon/runtime_.h"

namespace infini::ccl {

template <>
struct CclApi<BackendType::kCncl, Device::Type::kCambricon>
    : CnclApi<Device::Type::kCambricon> {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_CNCL_CAMBRICON_API_H_
