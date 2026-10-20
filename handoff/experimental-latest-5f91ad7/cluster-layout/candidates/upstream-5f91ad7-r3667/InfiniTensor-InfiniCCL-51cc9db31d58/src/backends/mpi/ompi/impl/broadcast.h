#ifndef INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_BROADCAST_H_
#define INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_BROADCAST_H_

#include <cstdlib>
#include <limits>

#include "backends/mpi/ompi/checks.h"
#include "backends/mpi/ompi/comm_instance.h"
#include "base/broadcast.h"
#include "communicator.h"
#include "data_type_impl.h"
#include "logging.h"
#include "runtime.h"

namespace infini::ccl {

template <Device::Type device_type>
class BroadcastImpl<BackendType::kOmpi, device_type> {
 public:
  static ReturnStatus Apply(const void *send_buff, void *recv_buff,
                            size_t count, DataType data_type, int root,
                            Communicator *comm, void *stream) {
    constexpr Device::Type kDev =
        ListGetBest<DevicePriority>(ActiveDevices<Broadcast>{});
    using Rt = Runtime<kDev>;

    auto *inst = static_cast<OmpiInstance *>(comm->inter_comm());
    size_t total_bytes = 0;
    ReturnStatus status = ValidateArgs(inst, count, data_type, total_bytes);
    if (status != ReturnStatus::kSuccess) {
      return status;
    }

    const int rank = comm->rank();
    const bool is_root = (rank == root);
    const bool is_out_of_place = (send_buff != recv_buff);

    // Resolve memory topology using the buffer guaranteed to be valid on all
    // nodes.
    const MemorySpace space = GetMemorySpace<kDev>(recv_buff);
    const bool is_device = (space == MemorySpace::kDevice);
    char *active_buf = nullptr;

    if (is_device) {
      active_buf = static_cast<char *>(std::malloc(total_bytes));
      if (!active_buf) {
        LOG("Failed to allocate host buffer for `Broadcast` staging.");
        return ReturnStatus::kSystemError;
      }

      if (is_root) {
        CHECK_STATUS(Rt, Rt::Memcpy(active_buf, send_buff, total_bytes,
                                    Rt::MemcpyDeviceToHost));
        CHECK_STATUS(Rt,
                     Rt::StreamSynchronize(static_cast<Rt::Stream>(stream)));
      }
    } else {
      if (is_root && is_out_of_place && recv_buff != nullptr) {
        std::memcpy(recv_buff, send_buff, total_bytes);
      }
      active_buf = static_cast<char *>(recv_buff);
    }

    size_t offset = 0;
    constexpr size_t kMaxMpiCount =
        static_cast<size_t>(std::numeric_limits<int>::max());

    while (offset < total_bytes) {
      size_t chunk = std::min(total_bytes - offset, kMaxMpiCount);
      INFINI_CHECK_MPI(MPI_Bcast(active_buf + offset, static_cast<int>(chunk),
                                 MPI_BYTE, root, inst->handle));
      offset += chunk;
    }

    if (is_device) {
      if (!is_root || is_out_of_place) {
        CHECK_STATUS(Rt, Rt::Memcpy(recv_buff, active_buf, total_bytes,
                                    Rt::MemcpyHostToDevice));
      }
      std::free(active_buf);
    }

    return ReturnStatus::kSuccess;
  }

 private:
  static ReturnStatus ValidateArgs(const OmpiInstance *inst, size_t count,
                                   DataType data_type,
                                   size_t &out_total_bytes) {
    if (!inst || inst->handle == MPI_COMM_NULL) {
      LOG("Invalid OpenMPI communicator instance for Broadcast.");
      return ReturnStatus::kInternalError;
    }

    size_t type_size = kDataTypeToSize.at(data_type);
    if (count > std::numeric_limits<size_t>::max() / type_size) {
      LOG("Broadcast byte size overflow.");
      return ReturnStatus::kInvalidArgument;
    }

    out_total_bytes = count * type_size;

    return ReturnStatus::kSuccess;
  }
};

template <>
struct BackendEnabled<Broadcast, BackendType::kOmpi> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_BROADCAST_H_
