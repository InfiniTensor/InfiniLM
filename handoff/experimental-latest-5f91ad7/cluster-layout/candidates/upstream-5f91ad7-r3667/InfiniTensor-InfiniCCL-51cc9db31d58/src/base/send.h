#ifndef INFINI_CCL_BASE_SEND_H_
#define INFINI_CCL_BASE_SEND_H_

#include "communicator.h"
#include "data_type_impl.h"
#include "logging.h"
#include "operation.h"
#include "return_status_impl.h"

namespace infini::ccl {

template <BackendType backend_type, Device::Type device_type>
struct SendImpl;

class Send : public Operation<Send> {
 public:
  template <BackendType backend_type, Device::Type device_type>
  static ReturnStatus Execute(const void *send_buff, size_t count,
                              DataType datatype, int peer, void *comm_handle,
                              void *stream) {
    if (!comm_handle) {
      LOG("Invalid communicator handle for `Send`.");
      return ReturnStatus::kInvalidArgument;
    }

    auto *comm = static_cast<Communicator *>(comm_handle);
    if (HasInvalidArgs(send_buff, count, datatype, peer, comm)) {
      return ReturnStatus::kInvalidArgument;
    }
    if (count == 0) {
      return ReturnStatus::kSuccess;
    }

    return SendImpl<backend_type, device_type>::Apply(
        send_buff, count, datatype, peer, comm, stream);
  }

 private:
  static bool HasInvalidArgs(const void *send_buff, size_t count,
                             DataType datatype, int peer, Communicator *comm) {
    if (datatype < DataType::kChar || datatype >= DataType::kNumTypes) {
      LOG("Invalid data type for `Send`.");
      return true;
    }
    if (peer < 0 || peer >= comm->size()) {
      LOG("Invalid peer rank for `Send`.");
      return true;
    }
    if (count == 0) {
      return false;
    }
    if (!send_buff) {
      LOG("Invalid send buffer pointer for `Send`.");
      return true;
    }
    return false;
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BASE_SEND_H_
