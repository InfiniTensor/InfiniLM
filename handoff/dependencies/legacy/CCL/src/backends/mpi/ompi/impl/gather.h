#ifndef INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_GATHER_H_
#define INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_GATHER_H_

#include <cstdlib>
#include <limits>

#include "backends/mpi/ompi/checks.h"
#include "backends/mpi/ompi/comm_instance.h"
#include "base/gather.h"
#include "communicator.h"
#include "data_type_impl.h"
#include "logging.h"
#include "runtime.h"

namespace infini::ccl {

template <Device::Type device_type>
class GatherImpl<BackendType::kOmpi, device_type> {
 public:
  static ReturnStatus Apply(const void *send_buff, void *recv_buff,
                            size_t count, DataType data_type, int root,
                            Communicator *comm, void *stream) {
    constexpr Device::Type kDev =
        ListGetBest<DevicePriority>(ActiveDevices<Gather>{});
    using Rt = Runtime<kDev>;

    auto *inst = static_cast<OmpiInstance *>(comm->inter_comm());
    if (!inst || inst->handle == MPI_COMM_NULL) {
      LOG("Invalid OpenMPI communicator instance for `Gather`.");
      return ReturnStatus::kInternalError;
    }

    size_t type_size = kDataTypeToSize.at(data_type);
    if (count > std::numeric_limits<size_t>::max() / type_size) {
      LOG("Byte size overflow for `Gather`.");
      return ReturnStatus::kInvalidArgument;
    }
    size_t send_bytes = count * type_size;
    size_t recv_bytes = send_bytes * static_cast<size_t>(comm->size());
    const bool is_root = comm->rank() == root;

    // Transfer raw bytes so the gather is correct for every data type,
    // including `kFloat16` / `kBFloat16`, which map to `MPI_BYTE`.
    if (send_bytes > static_cast<size_t>(std::numeric_limits<int>::max())) {
      LOG("Per-rank byte count exceeds MPI int range for `Gather`.");
      return ReturnStatus::kInvalidArgument;
    }
    int mpi_byte_count = static_cast<int>(send_bytes);

    // Host staging buffers. Only `root` allocates the receive side, since
    // `MPI_Gather` writes the gathered result only on `root`.
    void *host_sendbuf = std::malloc(send_bytes);
    void *host_recvbuf = is_root ? std::malloc(recv_bytes) : nullptr;
    if (!host_sendbuf || (is_root && !host_recvbuf)) {
      std::free(host_sendbuf);
      std::free(host_recvbuf);
      LOG("Failed to allocate host buffers for `Gather` staging.");
      return ReturnStatus::kSystemError;
    }

    CHECK_STATUS(Rt, Rt::Memcpy(host_sendbuf, send_buff, send_bytes,
                                Rt::MemcpyDeviceToHost));
    CHECK_STATUS(Rt, Rt::StreamSynchronize(static_cast<Rt::Stream>(stream)));

    // Note: `MPI_Gather`'s `recvcount` is the per-rank count, not the total.
    INFINI_CHECK_MPI(MPI_Gather(host_sendbuf, mpi_byte_count, MPI_BYTE,
                                host_recvbuf, mpi_byte_count, MPI_BYTE, root,
                                inst->handle));

    if (is_root) {
      CHECK_STATUS(Rt, Rt::Memcpy(recv_buff, host_recvbuf, recv_bytes,
                                  Rt::MemcpyHostToDevice));
    }

    std::free(host_sendbuf);
    std::free(host_recvbuf);

    return ReturnStatus::kSuccess;
  }
};

template <>
struct BackendEnabled<Gather, BackendType::kOmpi> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_GATHER_H_
