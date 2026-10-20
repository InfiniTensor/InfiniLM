#ifndef INFINI_CCL_BACKENDS_CCL_NCCL_IMPL_ALL_GATHER_H_
#define INFINI_CCL_BACKENDS_CCL_NCCL_IMPL_ALL_GATHER_H_

#include "backends/ccl/common/impl/all_gather.h"

namespace infini::ccl {

template <Device::Type device>
class AllGatherImpl<BackendType::kNccl, device>
    : public CclAllGatherImpl<BackendType::kNccl, device> {};

template <>
struct BackendEnabled<AllGather, BackendType::kNccl> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_NCCL_IMPL_ALL_GATHER_H_
