#ifndef INFINI_CCL_BACKENDS_CCL_COMMON_IMPL_COMM_INIT_RANK_H_
#define INFINI_CCL_BACKENDS_CCL_COMMON_IMPL_COMM_INIT_RANK_H_

#include <memory>

#include "backends/ccl/common/api.h"
#include "backends/ccl/common/comm_instance.h"
#include "base/comm_init_rank.h"
#include "communicator.h"
#include "logging.h"

namespace infini::ccl {

template <BackendType backend, Device::Type device>
class CclCommInitRankImpl {
 public:
  static ReturnStatus Apply(Communicator *comm, int nranks,
                            infinicclUniqueId id, int rank) {
    using Api = CclApi<backend, device>;
    using CommInstance = CclCommInstance<Api>;

    const auto *backend_id =
        reinterpret_cast<const typename Api::UniqueId *>(id.internal);

    typename Api::Comm ccl_handle{};
    auto status =
        Api::Check(Api::CommInitRank(&ccl_handle, nranks, *backend_id, rank));
    if (status != ReturnStatus::kSuccess) {
      return status;
    }

    comm->set_world_info(rank, nranks);

    auto inst = std::make_unique<CommInstance>();
    inst->handle = ccl_handle;

    comm->set_intra_comm(std::move(inst));

    return ReturnStatus::kSuccess;
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_COMMON_IMPL_COMM_INIT_RANK_H_
