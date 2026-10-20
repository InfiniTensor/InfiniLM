#ifndef INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_COMM_INIT_ALL_H_
#define INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_COMM_INIT_ALL_H_

#include "backends/mpi/ompi/checks.h"
#include "backends/mpi/ompi/comm_instance.h"
#include "base/comm_init_all.h"
#include "communicator.h"
#include "logging.h"

namespace infini::ccl {

template <Device::Type device_type>
class CommInitAllImpl<BackendType::kOmpi, device_type> {
 public:
  static ReturnStatus Apply(Communicator *comm, int n_dev,
                            const int *dev_list) {
    constexpr Device::Type kDev =
        ListGetBest<DevicePriority>(ActiveDevices<CommInitAll>{});
    using Rt = Runtime<kDev>;

    if (!comm) {
      // TODO(lzm): change to use `glog`.
      LOG("Failed to initialize OpenMPI communicator: invalid "
          "communicator pointer.");
      return ReturnStatus::kInternalError;
    }

    int rank, size;
    auto inst = std::make_unique<OmpiInstance>();
    INFINI_CHECK_MPI(MPI_Comm_dup(MPI_COMM_WORLD, &inst->handle));
    INFINI_CHECK_MPI(MPI_Comm_rank(inst->handle, &rank));
    INFINI_CHECK_MPI(MPI_Comm_size(inst->handle, &size));

    comm->set_world_info(rank, size);
    comm->set_inter_comm(std::move(inst));

    int local_rank = 0;
    char *local_rank_str = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (local_rank_str) {
      local_rank = atoi(local_rank_str);
    }

    // Use device from devlist or local_rank
    if (dev_list && local_rank < n_dev) {
      comm->set_device_id(dev_list[local_rank]);
    } else {
      comm->set_device_id(local_rank);
    }

    CHECK_STATUS(Rt, Rt::SetDevice(comm->device_id()));

    return ReturnStatus::kSuccess;
  }
};

template <>
struct BackendEnabled<CommInitAll, BackendType::kOmpi> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_COMM_INIT_ALL_H_
