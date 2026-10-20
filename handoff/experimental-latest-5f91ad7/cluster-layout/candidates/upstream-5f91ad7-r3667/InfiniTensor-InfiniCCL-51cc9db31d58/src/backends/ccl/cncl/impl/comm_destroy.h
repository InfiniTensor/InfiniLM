#ifndef INFINI_CCL_BACKENDS_CCL_CNCL_IMPL_COMM_DESTROY_H_
#define INFINI_CCL_BACKENDS_CCL_CNCL_IMPL_COMM_DESTROY_H_

#include "backends/ccl/common/impl/comm_destroy.h"

namespace infini::ccl {

template <Device::Type device>
class CommDestroyImpl<BackendType::kCncl, device>
    : public CclCommDestroyImpl<BackendType::kCncl, device> {};

template <>
struct BackendEnabled<CommDestroy, BackendType::kCncl> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_CNCL_IMPL_COMM_DESTROY_H_
