#ifndef INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_SEND_H_
#define INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_SEND_H_

#include <cstdlib>
#include <limits>

#include "backends/mpi/ompi/checks.h"
#include "backends/mpi/ompi/comm_instance.h"
#include "base/send.h"
#include "communicator.h"
#include "data_type_impl.h"
#include "logging.h"

namespace infini::ccl {

template <Device::Type device_type>
class SendImpl<BackendType::kOmpi, device_type> {
 public:
  static ReturnStatus Apply(const void *send_buff, size_t count,
                            DataType data_type, int peer, Communicator *comm,
                            void *stream) {
    constexpr Device::Type kDev =
        ListGetBest<DevicePriority>(ActiveDevices<Send>{});
    using Rt = Runtime<kDev>;

    auto *inst = static_cast<OmpiInstance *>(comm->inter_comm());
    if (!inst || inst->handle == MPI_COMM_NULL) {
      LOG("Invalid `OpenMPI` communicator instance for `Send`.");
      return ReturnStatus::kInternalError;
    }

    size_t type_size = kDataTypeToSize.at(data_type);
    if (count > std::numeric_limits<size_t>::max() / type_size) {
      LOG("Byte size overflow for `Send`.");
      return ReturnStatus::kInvalidArgument;
    }

    size_t total_bytes = count * type_size;
    void *host_buf = std::malloc(total_bytes);
    if (!host_buf) {
      LOG("Failed to allocate host buffer for `Send` staging.");
      return ReturnStatus::kSystemError;
    }

    CHECK_STATUS(Rt, Rt::Memcpy(host_buf, send_buff, total_bytes,
                                Rt::MemcpyDeviceToHost));
    CHECK_STATUS(Rt, Rt::StreamSynchronize(static_cast<Rt::Stream>(stream)));

    auto *bytes = static_cast<char *>(host_buf);
    size_t offset = 0;
    constexpr size_t kMaxMpiCount =
        static_cast<size_t>(std::numeric_limits<int>::max());
    constexpr int kTag = 0;
    while (offset < total_bytes) {
      size_t chunk = total_bytes - offset;
      if (chunk > kMaxMpiCount) {
        chunk = kMaxMpiCount;
      }
      INFINI_CHECK_MPI(MPI_Send(bytes + offset, static_cast<int>(chunk),
                                MPI_BYTE, peer, kTag, inst->handle));
      offset += chunk;
    }

    std::free(host_buf);
    return ReturnStatus::kSuccess;
  }
};

template <>
struct BackendEnabled<Send, BackendType::kOmpi> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_SEND_H_
