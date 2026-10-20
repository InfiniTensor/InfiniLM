#ifndef INFINI_CCL_BASE_BROADCAST_H_
#define INFINI_CCL_BASE_BROADCAST_H_

#include "communicator.h"
#include "data_type_impl.h"
#include "logging.h"
#include "operation.h"
#include "return_status_impl.h"

namespace infini::ccl {

template <BackendType backend_type, Device::Type device_type>
struct BroadcastImpl;

class Broadcast : public Operation<Broadcast> {
 public:
  template <BackendType backend_type, Device::Type device_type>
  static ReturnStatus Execute(const void *send_buff, void *recv_buff,
                              size_t count, DataType datatype, int root,
                              void *comm_handle, void *stream) {
    if (!comm_handle) {
      LOG("Invalid communicator handle for `Broadcast`.");
      return ReturnStatus::kInvalidArgument;
    }

    auto *comm = static_cast<Communicator *>(comm_handle);
    if (HasInvalidArgs(send_buff, recv_buff, count, datatype, root, comm)) {
      return ReturnStatus::kInvalidArgument;
    }
    if (count == 0) {
      return ReturnStatus::kSuccess;
    }

    return BroadcastImpl<backend_type, device_type>::Apply(
        send_buff, recv_buff, count, datatype, root, comm, stream);
  }

 private:
  static bool HasInvalidArgs(const void *send_buff, void *recv_buff,
                             size_t count, DataType datatype, int root,
                             Communicator *comm) {
    if (datatype < DataType::kChar || datatype >= DataType::kNumTypes) {
      LOG("Invalid data type for `Broadcast`.");
      return true;
    }
    if (root < 0 || root >= comm->size()) {
      LOG("Invalid root rank for `Broadcast`.");
      return true;
    }
    if (!recv_buff) {
      LOG("Invalid receive buffer pointer for `Broadcast`.");
      return true;
    }
    if (comm->rank() == root && !send_buff) {
      LOG("Invalid root send buffer pointer for `Broadcast`.");
      return true;
    }
    return false;
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BASE_BROADCAST_H_
