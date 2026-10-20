#ifndef INFINI_CCL_BACKENDS_CCL_NCCL_IMPL_GET_UNIQUE_ID_H_
#define INFINI_CCL_BACKENDS_CCL_NCCL_IMPL_GET_UNIQUE_ID_H_

#include "backends/ccl/common/impl/get_unique_id.h"

namespace infini::ccl {

template <Device::Type device>
class GetUniqueIdImpl<BackendType::kNccl, device>
    : public CclGetUniqueIdImpl<BackendType::kNccl, device> {};

template <>
struct BackendEnabled<GetUniqueId, BackendType::kNccl> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_NCCL_IMPL_GET_UNIQUE_ID_H_
