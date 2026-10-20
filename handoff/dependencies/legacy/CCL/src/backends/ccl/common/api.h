#ifndef INFINI_CCL_BACKENDS_CCL_COMMON_API_H_
#define INFINI_CCL_BACKENDS_CCL_COMMON_API_H_

#include "backend.h"
#include "comm_impl.h"
#include "device.h"

namespace infini::ccl {

template <BackendType backend, Device::Type device>
struct CclApi;

template <BackendType backend, Device::Type device>
struct CclTypeMap;

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_COMMON_API_H_
