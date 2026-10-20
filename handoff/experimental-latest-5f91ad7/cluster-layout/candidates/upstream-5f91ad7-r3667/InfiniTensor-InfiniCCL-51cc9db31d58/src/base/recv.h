#ifndef INFINI_CCL_BASE_RECV_H_
#define INFINI_CCL_BASE_RECV_H_

#include "communicator.h"
#include "data_type_impl.h"
#include "logging.h"
#include "operation.h"
#include "return_status_impl.h"

namespace infini::ccl {

template <BackendType backend_type, Device::Type device_type>
struct RecvImpl;

class Recv : public Operation<Recv> {
 public:
  template <BackendType backend_type, Device::Type device_type>
  static ReturnStatus Execute(void *recv_buff, size_t count, DataType datatype,
                              int peer, void *comm_handle, void *stream) {
    if (!comm_handle) {
      LOG("Invalid communicator handle for `Recv`.");
      return ReturnStatus::kInvalidArgument;
    }

    auto *comm = static_cast<Communicator *>(comm_handle);
    if (HasInvalidArgs(recv_buff, count, datatype, peer, comm)) {
      return ReturnStatus::kInvalidArgument;
    }
    if (count == 0) {
      return ReturnStatus::kSuccess;
    }

    return RecvImpl<backend_type, device_type>::Apply(
        recv_buff, count, datatype, peer, comm, stream);
  }

 private:
  static bool HasInvalidArgs(const void *recv_buff, size_t count,
                             DataType datatype, int peer, Communicator *comm) {
    if (datatype < DataType::kChar || datatype >= DataType::kNumTypes) {
      LOG("Invalid data type for `Recv`.");
      return true;
    }
    if (peer < 0 || peer >= comm->size()) {
      LOG("Invalid peer rank for `Recv`.");
      return true;
    }
    if (count == 0) {
      return false;
    }
    if (!recv_buff) {
      LOG("Invalid receive buffer pointer for `Recv`.");
      return true;
    }
    return false;
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BASE_RECV_H_
