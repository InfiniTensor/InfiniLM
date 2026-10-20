#ifndef INFINI_CCL_BASE_ALL_TO_ALL_H_
#define INFINI_CCL_BASE_ALL_TO_ALL_H_

#include "comm_impl.h"
#include "communicator.h"
#include "logging.h"
#include "operation.h"
#include "return_status_impl.h"

namespace infini::ccl {

template <BackendType backend_type, Device::Type device_type>
struct AllToAllImpl;

class AllToAll : public Operation<AllToAll> {
 public:
  template <BackendType backend_type, Device::Type device_type,
            typename... Args>
  static ReturnStatus Execute(const void *send_buff, void *recv_buff,
                              size_t count, DataType datatype,
                              void *comm_handle, void *stream) {
    if (HasInvalidArgs(send_buff, recv_buff, datatype, comm_handle)) {
      return ReturnStatus::kInvalidArgument;
    }
    auto *comm = static_cast<Communicator *>(comm_handle);
    return AllToAllImpl<backend_type, device_type>::Apply(
        send_buff, recv_buff, count, datatype, comm, stream);
  }

 private:
  static bool HasInvalidArgs(const void *send_buff, void *recv_buff,
                             DataType datatype, void *comm_handle) {
    if (!comm_handle) {
      // TODO(lzm): change to use `glog`.
      LOG("Invalid communicator handle for `AllToAll`.");
      return true;
    }
    if (!send_buff || !recv_buff) {
      LOG("Invalid buffer pointer for `AllToAll`.");
      return true;
    }
    if (datatype < DataType::kChar || datatype >= DataType::kNumTypes) {
      LOG("Invalid data type for `AllToAll`.");
      return true;
    }
    return false;
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BASE_ALL_TO_ALL_H_
