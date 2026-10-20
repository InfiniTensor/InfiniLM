#ifndef INFINI_CCL_BACKENDS_CCL_COMMON_IMPL_GET_UNIQUE_ID_H_
#define INFINI_CCL_BACKENDS_CCL_COMMON_IMPL_GET_UNIQUE_ID_H_

#include <cstring>

#include "backends/ccl/common/api.h"
#include "base/get_unique_id.h"

namespace infini::ccl {

template <BackendType backend, Device::Type device>
class CclGetUniqueIdImpl {
 public:
  static ReturnStatus Apply(infinicclUniqueId *id) {
    using Api = CclApi<backend, device>;

    typename Api::UniqueId ccl_id;

    auto status = Api::Check(Api::GetUniqueId(&ccl_id));
    if (status != ReturnStatus::kSuccess) {
      return status;
    }

    // Safety check: ensure our public struct is large enough.
    static_assert(
        sizeof(typename Api::UniqueId) <= sizeof(id->internal),
        "`infinicclUniqueId::internal` is too small for backend unique id");

    std::memcpy(id->internal, &ccl_id, sizeof(ccl_id));

    return ReturnStatus::kSuccess;
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_COMMON_IMPL_GET_UNIQUE_ID_H_
