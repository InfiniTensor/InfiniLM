#ifndef INFINI_CCL_BACKENDS_CCL_CNCL_IMPL_RECV_H_
#define INFINI_CCL_BACKENDS_CCL_CNCL_IMPL_RECV_H_

#include "backends/ccl/common/impl/recv.h"

namespace infini::ccl {

template <Device::Type device>
class RecvImpl<BackendType::kCncl, device>
    : public CclRecvImpl<BackendType::kCncl, device> {};

template <>
struct BackendEnabled<Recv, BackendType::kCncl> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_CNCL_IMPL_RECV_H_
