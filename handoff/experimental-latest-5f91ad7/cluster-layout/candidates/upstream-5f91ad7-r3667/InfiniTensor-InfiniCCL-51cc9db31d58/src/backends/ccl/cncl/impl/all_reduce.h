#ifndef INFINI_CCL_BACKENDS_CCL_CNCL_IMPL_ALL_REDUCE_H_
#define INFINI_CCL_BACKENDS_CCL_CNCL_IMPL_ALL_REDUCE_H_

#include "backends/ccl/common/impl/all_reduce.h"

namespace infini::ccl {

template <Device::Type device>
class AllReduceImpl<BackendType::kCncl, device>
    : public CclAllReduceImpl<BackendType::kCncl, device> {};

template <>
struct BackendEnabled<AllReduce, BackendType::kCncl> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_CNCL_IMPL_ALL_REDUCE_H_
