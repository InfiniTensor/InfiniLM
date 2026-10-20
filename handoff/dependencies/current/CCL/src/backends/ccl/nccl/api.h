#ifndef INFINI_CCL_BACKENDS_CCL_NCCL_API_H_
#define INFINI_CCL_BACKENDS_CCL_NCCL_API_H_

#include <cstddef>

#include "backends/ccl/common/api.h"
#include "backends/ccl/nccl/compat.h"
#include "logging.h"
#include "return_status_impl.h"
#include "runtime.h"

namespace infini::ccl {

template <Device::Type device>
struct NcclDataTypeTraits;

template <Device::Type device>
struct NcclApi {
  static constexpr BackendType kBackendType = BackendType::kNccl;
  static constexpr Device::Type kDeviceType = device;

  using Comm = ncclComm_t;
  using UniqueId = ncclUniqueId;
  using Result = ncclResult_t;
  using DataType = ncclDataType_t;
  using RedOp = ncclRedOp_t;
  using Stream = typename Runtime<device>::Stream;

  static ReturnStatus Check(Result result) {
    if (result != ncclSuccess) {
      LOG(ncclGetErrorString(result));
      return ReturnStatus::kSystemError;
    }
    return ReturnStatus::kSuccess;
  }

  static Result GetUniqueId(UniqueId* id) { return ncclGetUniqueId(id); }

  static Result CommInitRank(Comm* comm, int nranks, UniqueId id, int rank) {
    return ncclCommInitRank(comm, nranks, id, rank);
  }

  static Result CommDestroy(Comm comm) { return ncclCommDestroy(comm); }

  static Result AllReduce(const void* send_buff, void* recv_buff, size_t count,
                          DataType data_type, RedOp op, Comm comm,
                          Stream stream) {
    return ncclAllReduce(send_buff, recv_buff, count, data_type, op, comm,
                         stream);
  }

  static Result AllGather(const void* send_buff, void* recv_buff,
                          size_t send_count, DataType data_type, Comm comm,
                          Stream stream) {
    return ncclAllGather(send_buff, recv_buff, send_count, data_type, comm,
                         stream);
  }

  static Result ReduceScatter(const void* send_buff, void* recv_buff,
                              size_t recv_count, DataType data_type, RedOp op,
                              Comm comm, Stream stream) {
    return ncclReduceScatter(send_buff, recv_buff, recv_count, data_type, op,
                             comm, stream);
  }

  static Result Send(const void* send_buff, size_t count, DataType data_type,
                     int peer, Comm comm, Stream stream) {
    return ncclSend(send_buff, count, data_type, peer, comm, stream);
  }

  static Result Recv(void* recv_buff, size_t count, DataType data_type,
                     int peer, Comm comm, Stream stream) {
    return ncclRecv(recv_buff, count, data_type, peer, comm, stream);
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_NCCL_API_H_
