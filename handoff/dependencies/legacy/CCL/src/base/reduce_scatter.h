#ifndef INFINI_CCL_BASE_REDUCE_SCATTER_H_
#define INFINI_CCL_BASE_REDUCE_SCATTER_H_

#include "comm_impl.h"
#include "communicator.h"
#include "logging.h"
#include "operation.h"
#include "return_status_impl.h"

namespace infini::ccl {

template <BackendType backend_type, Device::Type device_type>
struct ReduceScatterImpl;

class ReduceScatter : public Operation<ReduceScatter> {
 public:
  template <BackendType backend_type, Device::Type device_type,
            typename... Args>
  static ReturnStatus Execute(const void *send_buff, void *recv_buff,
                              size_t recv_count, DataType datatype,
                              ReductionOpType op, void *comm_handle,
                              void *stream) {
    if (HasInvalidArgs(send_buff, recv_buff, datatype, op, comm_handle)) {
      return ReturnStatus::kInvalidArgument;
    }
    auto *comm = static_cast<Communicator *>(comm_handle);
    return ReduceScatterImpl<backend_type, device_type>::Apply(
        send_buff, recv_buff, recv_count, datatype, op, comm, stream);
  }

 private:
  static bool HasInvalidArgs(const void *send_buff, void *recv_buff,
                             DataType datatype, ReductionOpType op,
                             void *comm_handle) {
    if (!comm_handle) {
      // TODO(lzm): change to use `glog`.
      LOG("Invalid communicator handle for `ReduceScatter`.");
      return true;
    }
    if (!send_buff || !recv_buff) {
      LOG("Invalid buffer pointer for `ReduceScatter`.");
      return true;
    }
    if (op < ReductionOpType::kSum || op >= ReductionOpType::kNumRedOps) {
      LOG("Invalid reduction operation for `ReduceScatter`.");
      return true;
    }
    if (datatype < DataType::kChar || datatype >= DataType::kNumTypes) {
      LOG("Invalid data type for `ReduceScatter`.");
      return true;
    }
    return false;
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BASE_REDUCE_SCATTER_H_
