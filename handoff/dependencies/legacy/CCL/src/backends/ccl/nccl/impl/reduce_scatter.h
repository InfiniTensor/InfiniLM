#ifndef INFINI_CCL_BACKENDS_CCL_NCCL_IMPL_REDUCE_SCATTER_H_
#define INFINI_CCL_BACKENDS_CCL_NCCL_IMPL_REDUCE_SCATTER_H_

#include "backends/ccl/common/impl/reduce_scatter.h"

namespace infini::ccl {

template <Device::Type device>
class ReduceScatterImpl<BackendType::kNccl, device>
    : public CclReduceScatterImpl<BackendType::kNccl, device> {};

template <>
struct BackendEnabled<ReduceScatter, BackendType::kNccl> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_NCCL_IMPL_REDUCE_SCATTER_H_
