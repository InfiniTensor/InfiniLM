#ifndef INFINI_CCL_BACKENDS_CCL_NCCL_HYGON_API_H_
#define INFINI_CCL_BACKENDS_CCL_NCCL_HYGON_API_H_

#include "backends/ccl/nccl/api.h"
#include "devices/hygon/runtime_.h"

namespace infini::ccl {

template <>
struct CclApi<BackendType::kNccl, Device::Type::kHygon>
    : NcclApi<Device::Type::kHygon> {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_NCCL_HYGON_API_H_
