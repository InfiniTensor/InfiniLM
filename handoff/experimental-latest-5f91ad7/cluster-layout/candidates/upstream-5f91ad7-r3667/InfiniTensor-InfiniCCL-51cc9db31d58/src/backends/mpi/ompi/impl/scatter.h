#ifndef INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_SCATTER_H_
#define INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_SCATTER_H_

#include <cstdlib>
#include <limits>

#include "backends/mpi/ompi/checks.h"
#include "backends/mpi/ompi/comm_instance.h"
#include "base/scatter.h"
#include "communicator.h"
#include "data_type_impl.h"
#include "logging.h"
#include "runtime.h"

namespace infini::ccl {

template <Device::Type device_type>
class ScatterImpl<BackendType::kOmpi, device_type> {
 public:
  static ReturnStatus Apply(const void *send_buff, void *recv_buff,
                            size_t count, DataType data_type, int root,
                            Communicator *comm, void *stream) {
    constexpr Device::Type kDev =
        ListGetBest<DevicePriority>(ActiveDevices<Scatter>{});
    using Rt = Runtime<kDev>;

    auto *inst = static_cast<OmpiInstance *>(comm->inter_comm());
    if (!inst || inst->handle == MPI_COMM_NULL) {
      LOG("Invalid OpenMPI communicator instance for `Scatter`.");
      return ReturnStatus::kInternalError;
    }

    size_t type_size = kDataTypeToSize.at(data_type);
    if (count > std::numeric_limits<size_t>::max() / type_size) {
      LOG("Byte size overflow for `Scatter`.");
      return ReturnStatus::kInvalidArgument;
    }
    size_t recv_bytes = count * type_size;
    size_t send_bytes = recv_bytes * static_cast<size_t>(comm->size());
    const bool is_root = comm->rank() == root;

    // Transfer raw bytes so the scatter is correct for every data type,
    // including `kFloat16` / `kBFloat16`, which map to `MPI_BYTE`.
    if (recv_bytes > static_cast<size_t>(std::numeric_limits<int>::max())) {
      LOG("Per-rank byte count exceeds MPI int range for `Scatter`.");
      return ReturnStatus::kInvalidArgument;
    }
    int mpi_byte_count = static_cast<int>(recv_bytes);

    // Host staging buffers. Only `root` allocates the send side, since
    // `MPI_Scatter` reads the distributed blocks only from `root`.
    void *host_sendbuf = is_root ? std::malloc(send_bytes) : nullptr;
    void *host_recvbuf = std::malloc(recv_bytes);
    if ((is_root && !host_sendbuf) || !host_recvbuf) {
      std::free(host_sendbuf);
      std::free(host_recvbuf);
      LOG("Failed to allocate host buffers for `Scatter` staging.");
      return ReturnStatus::kSystemError;
    }

    if (is_root) {
      CHECK_STATUS(Rt, Rt::Memcpy(host_sendbuf, send_buff, send_bytes,
                                  Rt::MemcpyDeviceToHost));
    }
    CHECK_STATUS(Rt, Rt::StreamSynchronize(static_cast<Rt::Stream>(stream)));

    // Note: `MPI_Scatter`'s `sendcount` is the per-rank count, not the total.
    INFINI_CHECK_MPI(MPI_Scatter(host_sendbuf, mpi_byte_count, MPI_BYTE,
                                 host_recvbuf, mpi_byte_count, MPI_BYTE, root,
                                 inst->handle));

    CHECK_STATUS(Rt, Rt::Memcpy(recv_buff, host_recvbuf, recv_bytes,
                                Rt::MemcpyHostToDevice));

    std::free(host_sendbuf);
    std::free(host_recvbuf);

    return ReturnStatus::kSuccess;
  }
};

template <>
struct BackendEnabled<Scatter, BackendType::kOmpi> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_SCATTER_H_
