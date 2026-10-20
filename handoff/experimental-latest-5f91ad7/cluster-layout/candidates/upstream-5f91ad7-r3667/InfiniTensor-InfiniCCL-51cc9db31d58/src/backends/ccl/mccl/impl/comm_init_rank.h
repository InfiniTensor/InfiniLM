#ifndef INFINI_CCL_BACKENDS_CCL_MCCL_IMPL_COMM_INIT_RANK_H_
#define INFINI_CCL_BACKENDS_CCL_MCCL_IMPL_COMM_INIT_RANK_H_

#include "backends/ccl/common/impl/comm_init_rank.h"

namespace infini::ccl {

template <Device::Type device>
class CommInitRankImpl<BackendType::kMccl, device>
    : public CclCommInitRankImpl<BackendType::kMccl, device> {};

template <>
struct BackendEnabled<CommInitRank, BackendType::kMccl> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_MCCL_IMPL_COMM_INIT_RANK_H_
