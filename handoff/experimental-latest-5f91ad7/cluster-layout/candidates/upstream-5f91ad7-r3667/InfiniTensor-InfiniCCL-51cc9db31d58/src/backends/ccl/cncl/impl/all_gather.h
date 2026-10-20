#ifndef INFINI_CCL_BACKENDS_CCL_CNCL_IMPL_ALL_GATHER_H_
#define INFINI_CCL_BACKENDS_CCL_CNCL_IMPL_ALL_GATHER_H_

#include "backends/ccl/common/impl/all_gather.h"

namespace infini::ccl {

template <Device::Type device>
class AllGatherImpl<BackendType::kCncl, device>
    : public CclAllGatherImpl<BackendType::kCncl, device> {};

template <>
struct BackendEnabled<AllGather, BackendType::kCncl> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_CNCL_IMPL_ALL_GATHER_H_
