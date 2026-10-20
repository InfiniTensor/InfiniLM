#ifndef INFINI_CCL_BACKENDS_CCL_MCCL_API_H_
#define INFINI_CCL_BACKENDS_CCL_MCCL_API_H_

#include <mccl.h>

#include <cstddef>

#include "backends/ccl/common/api.h"
#include "logging.h"
#include "return_status_impl.h"
#include "runtime.h"

namespace infini::ccl {

template <Device::Type device>
struct McclDataTypeTraits;

template <Device::Type device>
struct McclApi {
  static constexpr BackendType kBackendType = BackendType::kMccl;
  static constexpr Device::Type kDeviceType = device;

  using Comm = mcclComm_t;
  using UniqueId = mcclUniqueId;
  using Result = mcclResult_t;
  using DataType = mcclDataType_t;
  using RedOp = mcclRedOp_t;
  using Stream = typename Runtime<device>::Stream;

  static ReturnStatus Check(Result result) {
    if (result != mcclSuccess) {
      LOG(mcclGetErrorString(result));
      return ReturnStatus::kSystemError;
    }
    return ReturnStatus::kSuccess;
  }

  static Result GetUniqueId(UniqueId *id) { return mcclGetUniqueId(id); }

  static Result CommInitRank(Comm *comm, int nranks, UniqueId id, int rank) {
    return mcclCommInitRank(comm, nranks, id, rank);
  }

  static Result CommDestroy(Comm comm) { return mcclCommDestroy(comm); }

  static Result AllReduce(const void *send_buff, void *recv_buff, size_t count,
                          DataType data_type, RedOp op, Comm comm,
                          Stream stream) {
    return mcclAllReduce(send_buff, recv_buff, count, data_type, op, comm,
                         stream);
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_MCCL_API_H_
