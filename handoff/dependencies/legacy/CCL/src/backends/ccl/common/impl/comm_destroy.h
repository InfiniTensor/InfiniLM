#ifndef INFINI_CCL_BACKENDS_CCL_COMMON_IMPL_COMM_DESTROY_H_
#define INFINI_CCL_BACKENDS_CCL_COMMON_IMPL_COMM_DESTROY_H_

#include "backends/ccl/common/api.h"
#include "backends/ccl/common/comm_instance.h"
#include "base/comm_destroy.h"
#include "communicator.h"

namespace infini::ccl {

template <BackendType backend, Device::Type device>
class CclCommDestroyImpl {
 public:
  static ReturnStatus Apply(void *comm) {
    using Api = CclApi<backend, device>;
    using CommInstance = CclCommInstance<Api>;

    auto *comm_internal = static_cast<Communicator *>(comm);
    if (!comm_internal) {
      return ReturnStatus::kInternalError;
    }

    if (auto *intra =
            static_cast<CommInstance *>(comm_internal->intra_comm())) {
      intra->Destroy();
    }

    comm_internal->set_intra_comm(nullptr);

    return ReturnStatus::kSuccess;
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_COMMON_IMPL_COMM_DESTROY_H_
