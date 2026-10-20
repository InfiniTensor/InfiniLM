#ifndef INFINI_CCL_BACKENDS_CCL_NCCL_IMPL_SEND_H_
#define INFINI_CCL_BACKENDS_CCL_NCCL_IMPL_SEND_H_

#include "backends/ccl/common/impl/send.h"

namespace infini::ccl {

template <Device::Type device>
class SendImpl<BackendType::kNccl, device>
    : public CclSendImpl<BackendType::kNccl, device> {};

template <>
struct BackendEnabled<Send, BackendType::kNccl> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_NCCL_IMPL_SEND_H_
