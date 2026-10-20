#ifndef INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_REDUCE_SCATTER_H_
#define INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_REDUCE_SCATTER_H_

#include <limits>
#include <type_traits>

#include "backends/mpi/ompi/checks.h"
#include "backends/mpi/ompi/comm_instance.h"
#include "backends/mpi/ompi/type_map.h"
#include "base/reduce_scatter.h"
#include "communicator.h"
#include "dispatcher.h"
#include "logging.h"

namespace infini::ccl {

template <Device::Type device_type>
class ReduceScatterImpl<BackendType::kOmpi, device_type> {
 public:
  static ReturnStatus Apply(const void *send_buff, void *recv_buff,
                            size_t recv_count, DataType data_type,
                            ReductionOpType op, Communicator *comm,
                            void *stream) {
    constexpr Device::Type kDev =
        ListGetBest<DevicePriority>(ActiveDevices<ReduceScatter>{});
    using Rt = Runtime<kDev>;

    auto *inst = static_cast<OmpiInstance *>(comm->inter_comm());

    if (!inst || inst->handle == MPI_COMM_NULL) {
      LOG("Invalid OpenMPI communicator instance for `ReduceScatter`.");
      return ReturnStatus::kInternalError;
    }

    MPI_Datatype mpi_type = DataTypeToOmpiType(data_type);
    MPI_Op mpi_op = RedOpToOmpiOp(op);

    size_t world_size = static_cast<size_t>(comm->size());
    size_t type_size = kDataTypeToSize.at(data_type);
    size_t send_count = recv_count * world_size;
    size_t send_bytes = send_count * type_size;
    size_t recv_bytes = recv_count * type_size;

    // Handle GPU Memory (Staging Pattern)
    // Note: we simply use host-staging for now.
    void *host_sendbuf = malloc(send_bytes);
    void *host_recvbuf = malloc(recv_bytes);
    if (!host_sendbuf || !host_recvbuf) {
      free(host_sendbuf);
      free(host_recvbuf);
      LOG("Failed to allocate host buffers for `ReduceScatter` staging.");
      return ReturnStatus::kSystemError;
    }

    CHECK_STATUS(Rt, Rt::Memcpy(host_sendbuf, send_buff, send_bytes,
                                Rt::MemcpyDeviceToHost));
    CHECK_STATUS(Rt, Rt::StreamSynchronize(static_cast<Rt::Stream>(stream)));

    if (recv_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
      LOG("recv_count exceeds MPI int range for `ReduceScatter`.");
      free(host_sendbuf);
      free(host_recvbuf);
      return ReturnStatus::kInvalidArgument;
    }
    int mpi_recv_count = static_cast<int>(recv_count);

    INFINI_CHECK_MPI(MPI_Reduce_scatter_block(host_sendbuf, host_recvbuf,
                                              mpi_recv_count, mpi_type, mpi_op,
                                              inst->handle));

    if (op == ReductionOpType::kAvg) {
      float scale = 1.0f / static_cast<float>(world_size);

      DispatchFunc<kDev, AllTypes>(data_type, [&](auto dtype) {
        using T = typename decltype(dtype)::type;

        T *typed_buf = static_cast<T *>(host_recvbuf);

        // Simply do the averaging on the CPU before the H2D copy.
        for (size_t i = 0; i < recv_count; ++i) {
          if constexpr (std::is_integral_v<T>) {
            // Scale in floating point first; casting `scale` to an integer
            // type would truncate it to `0` and zero out the result.
            typed_buf[i] =
                CastTo<kDev, T>(CastTo<kDev, double>(typed_buf[i]) * scale);
          } else if constexpr (SupportsOpValue<T, decltype(scale),
                                               MulAssignOp>) {
            typed_buf[i] *= scale;
          } else {
            float f_val = ToFloat<kDev>(typed_buf[i]) * scale;
            typed_buf[i] = CastTo<kDev, T>(f_val);
          }
        }
      });
    }

    CHECK_STATUS(Rt, Rt::Memcpy(recv_buff, host_recvbuf, recv_bytes,
                                Rt::MemcpyHostToDevice));

    free(host_sendbuf);
    free(host_recvbuf);

    return ReturnStatus::kSuccess;
  }
};

template <>
struct BackendEnabled<ReduceScatter, BackendType::kOmpi> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_REDUCE_SCATTER_H_
