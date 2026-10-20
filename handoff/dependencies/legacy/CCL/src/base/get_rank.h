#ifndef INFINI_CCL_BASE_GET_RANK_H_
#define INFINI_CCL_BASE_GET_RANK_H_

#include "operation.h"
#include "return_status_impl.h"

namespace infini::ccl {

template <BackendType backend_type, Device::Type device_type>
struct GetRankImpl;

class GetRank : public Operation<GetRank> {
 public:
  template <BackendType backend_type, Device::Type device_type,
            typename... Args>
  static ReturnStatus Execute(Args &&...args) {
    return GetRankImpl<backend_type, device_type>::Apply(
        std::forward<Args>(args)...);
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BASE_GET_RANK_H_
