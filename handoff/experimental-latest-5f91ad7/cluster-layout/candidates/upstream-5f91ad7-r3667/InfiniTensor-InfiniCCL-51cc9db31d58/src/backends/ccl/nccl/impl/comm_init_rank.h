#ifndef INFINI_CCL_BACKENDS_CCL_NCCL_IMPL_COMM_INIT_RANK_H_
#define INFINI_CCL_BACKENDS_CCL_NCCL_IMPL_COMM_INIT_RANK_H_

#include "backends/ccl/common/impl/comm_init_rank.h"

namespace infini::ccl {

template <Device::Type device>
class CommInitRankImpl<BackendType::kNccl, device>
    : public CclCommInitRankImpl<BackendType::kNccl, device> {};

template <>
struct BackendEnabled<CommInitRank, BackendType::kNccl> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_NCCL_IMPL_COMM_INIT_RANK_H_
