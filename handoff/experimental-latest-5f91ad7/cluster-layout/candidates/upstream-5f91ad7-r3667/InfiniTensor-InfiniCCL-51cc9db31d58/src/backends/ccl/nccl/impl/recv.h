#ifndef INFINI_CCL_BACKENDS_CCL_NCCL_IMPL_RECV_H_
#define INFINI_CCL_BACKENDS_CCL_NCCL_IMPL_RECV_H_

#include "backends/ccl/common/impl/recv.h"

namespace infini::ccl {

template <Device::Type device>
class RecvImpl<BackendType::kNccl, device>
    : public CclRecvImpl<BackendType::kNccl, device> {};

template <>
struct BackendEnabled<Recv, BackendType::kNccl> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_NCCL_IMPL_RECV_H_
