#ifndef INFINI_CCL_BASE_ALL_GATHER_H_
#define INFINI_CCL_BASE_ALL_GATHER_H_

#include "comm_impl.h"
#include "communicator.h"
#include "logging.h"
#include "operation.h"
#include "return_status_impl.h"

namespace infini::ccl {

template <BackendType backend_type, Device::Type device_type>
struct AllGatherImpl;

class AllGather : public Operation<AllGather> {
 public:
  template <BackendType backend_type, Device::Type device_type,
            typename... Args>
  static ReturnStatus Execute(const void *send_buff, void *recv_buff,
                              size_t count, DataType datatype,
                              void *comm_handle, void *stream) {
    if (count == 0) {
      return ReturnStatus::kSuccess;
    }

    if (HasInvalidArgs(send_buff, recv_buff, datatype, comm_handle)) {
      return ReturnStatus::kInvalidArgument;
    }

    auto *comm = static_cast<Communicator *>(comm_handle);

    return AllGatherImpl<backend_type, device_type>::Apply(
        send_buff, recv_buff, count, datatype, comm, stream);
  }

 private:
  static bool HasInvalidArgs(const void *send_buff, void *recv_buff,
                             DataType datatype, void *comm_handle) {
    if (!comm_handle) {
      // TODO(lzm): change to use `glog`.
      LOG("Invalid communicator handle for `AllGather`.");
      return true;
    }
    if (!send_buff || !recv_buff) {
      LOG("Invalid buffer pointer for `AllGather`.");
      return true;
    }
    if (datatype < DataType::kChar || datatype >= DataType::kNumTypes) {
      LOG("Invalid data type for `AllGather`.");
      return true;
    }
    return false;
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BASE_ALL_GATHER_H_
