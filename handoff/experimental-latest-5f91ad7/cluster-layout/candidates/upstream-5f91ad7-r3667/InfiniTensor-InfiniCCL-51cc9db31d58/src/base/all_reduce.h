#ifndef INFINI_CCL_BASE_ALL_REDUCE_H_
#define INFINI_CCL_BASE_ALL_REDUCE_H_

#include "comm_impl.h"
#include "communicator.h"
#include "logging.h"
#include "operation.h"
#include "return_status_impl.h"

namespace infini::ccl {

template <BackendType backend_type, Device::Type device_type>
struct AllReduceImpl;

class AllReduce : public Operation<AllReduce> {
 public:
  template <BackendType backend_type, Device::Type device_type,
            typename... Args>
  static ReturnStatus Execute(const void *send_buff, void *recv_buff,
                              size_t count, DataType datatype,
                              ReductionOpType op, void *comm_handle,
                              void *stream) {
    if (HasInvalidArgs(send_buff, recv_buff, datatype, op, comm_handle)) {
      return ReturnStatus::kInvalidArgument;
    }
    auto *comm = static_cast<Communicator *>(comm_handle);
    return AllReduceImpl<backend_type, device_type>::Apply(
        send_buff, recv_buff, count, datatype, op, comm, stream);
  }

 private:
  static bool HasInvalidArgs(const void *send_buff, void *recv_buff,
                             DataType datatype, ReductionOpType op,
                             void *comm_handle) {
    if (!comm_handle) {
      // TODO(lzm): change to use `glog`.
      LOG("Invalid communicator handle for AllReduce.");
      return true;
    }
    if (!send_buff || !recv_buff) {
      LOG("Invalid buffer pointer for AllReduce.");
      return true;
    }
    if (op < ReductionOpType::kSum || op >= ReductionOpType::kNumRedOps) {
      LOG("Invalid reduction operation for AllReduce.");
      return true;
    }
    if (datatype < DataType::kChar || datatype >= DataType::kNumTypes) {
      LOG("Invalid data type for AllReduce.");
      return true;
    }
    return false;
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BASE_ALL_REDUCE_H_
