#ifndef INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_ALL_TO_ALL_H_
#define INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_ALL_TO_ALL_H_

#include <limits>

#include "backends/mpi/ompi/checks.h"
#include "backends/mpi/ompi/comm_instance.h"
#include "backends/mpi/ompi/type_map.h"
#include "base/all_to_all.h"
#include "communicator.h"
#include "dispatcher.h"
#include "logging.h"

namespace infini::ccl {

template <Device::Type device_type>
class AllToAllImpl<BackendType::kOmpi, device_type> {
 public:
  static ReturnStatus Apply(const void *send_buff, void *recv_buff,
                            size_t count, DataType data_type,
                            Communicator *comm, void *stream) {
    constexpr Device::Type kDev =
        ListGetBest<DevicePriority>(ActiveDevices<AllToAll>{});
    using Rt = Runtime<kDev>;

    auto *inst = static_cast<OmpiInstance *>(comm->inter_comm());

    if (!inst || inst->handle == MPI_COMM_NULL) {
      LOG("Invalid OpenMPI communicator instance for `AllToAll`.");
      return ReturnStatus::kInternalError;
    }

    if (count > static_cast<size_t>(std::numeric_limits<int>::max())) {
      LOG("count exceeds MPI `int` range for `AllToAll`.");
      return ReturnStatus::kInvalidArgument;
    }
    int mpi_count = static_cast<int>(count);

    size_t world_size = static_cast<size_t>(comm->size());
    MPI_Datatype mpi_type = DataTypeToOmpiType(data_type);
    size_t type_size = kDataTypeToSize.at(data_type);
    size_t total_count = count * world_size;
    size_t total_bytes = total_count * type_size;

    // Handle GPU Memory (Staging Pattern)
    // Note: we simply use host-staging for now.
    void *host_sendbuf = malloc(total_bytes);
    void *host_recvbuf = malloc(total_bytes);
    if (!host_sendbuf || !host_recvbuf) {
      free(host_sendbuf);
      free(host_recvbuf);
      LOG("Failed to allocate host buffers for `AllToAll` staging.");
      return ReturnStatus::kSystemError;
    }

    CHECK_STATUS(Rt, Rt::Memcpy(host_sendbuf, send_buff, total_bytes,
                                Rt::MemcpyDeviceToHost));

    CHECK_STATUS(Rt, Rt::StreamSynchronize(static_cast<Rt::Stream>(stream)));

    INFINI_CHECK_MPI(MPI_Alltoall(host_sendbuf, mpi_count, mpi_type,
                                  host_recvbuf, mpi_count, mpi_type,
                                  inst->handle));

    CHECK_STATUS(Rt, Rt::Memcpy(recv_buff, host_recvbuf, total_bytes,
                                Rt::MemcpyHostToDevice));

    free(host_sendbuf);
    free(host_recvbuf);
    return ReturnStatus::kSuccess;
  }
};

template <>
struct BackendEnabled<AllToAll, BackendType::kOmpi> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_ALL_TO_ALL_H_
