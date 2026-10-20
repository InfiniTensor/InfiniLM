#ifndef INFINI_CCL_BACKENDS_CCL_CNCL_IMPL_GET_UNIQUE_ID_H_
#define INFINI_CCL_BACKENDS_CCL_CNCL_IMPL_GET_UNIQUE_ID_H_

#include "backends/ccl/common/impl/get_unique_id.h"

namespace infini::ccl {

template <Device::Type device>
class GetUniqueIdImpl<BackendType::kCncl, device>
    : public CclGetUniqueIdImpl<BackendType::kCncl, device> {};

template <>
struct BackendEnabled<GetUniqueId, BackendType::kCncl> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_CNCL_IMPL_GET_UNIQUE_ID_H_
