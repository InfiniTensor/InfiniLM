#ifndef INFINI_CCL_BACKENDS_CCL_NCCL_IMPL_COMM_DESTROY_H_
#define INFINI_CCL_BACKENDS_CCL_NCCL_IMPL_COMM_DESTROY_H_

#include "backends/ccl/common/impl/comm_destroy.h"

namespace infini::ccl {

template <Device::Type device>
class CommDestroyImpl<BackendType::kNccl, device>
    : public CclCommDestroyImpl<BackendType::kNccl, device> {};

template <>
struct BackendEnabled<CommDestroy, BackendType::kNccl> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_NCCL_IMPL_COMM_DESTROY_H_
