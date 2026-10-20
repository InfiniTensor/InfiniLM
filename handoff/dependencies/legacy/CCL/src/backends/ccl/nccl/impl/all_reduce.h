#ifndef INFINI_CCL_BACKENDS_CCL_NCCL_IMPL_ALL_REDUCE_H_
#define INFINI_CCL_BACKENDS_CCL_NCCL_IMPL_ALL_REDUCE_H_

#include "backends/ccl/common/impl/all_reduce.h"

namespace infini::ccl {

template <Device::Type device>
class AllReduceImpl<BackendType::kNccl, device>
    : public CclAllReduceImpl<BackendType::kNccl, device> {};

template <>
struct BackendEnabled<AllReduce, BackendType::kNccl> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_NCCL_IMPL_ALL_REDUCE_H_
