#ifndef INFINI_CCL_BACKENDS_CCL_MCCL_IMPL_COMM_DESTROY_H_
#define INFINI_CCL_BACKENDS_CCL_MCCL_IMPL_COMM_DESTROY_H_

#include "backends/ccl/common/impl/comm_destroy.h"

namespace infini::ccl {

template <Device::Type device>
class CommDestroyImpl<BackendType::kMccl, device>
    : public CclCommDestroyImpl<BackendType::kMccl, device> {};

template <>
struct BackendEnabled<CommDestroy, BackendType::kMccl> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_MCCL_IMPL_COMM_DESTROY_H_
