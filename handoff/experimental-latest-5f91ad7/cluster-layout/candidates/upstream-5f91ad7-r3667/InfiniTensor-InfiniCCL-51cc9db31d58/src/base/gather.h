#ifndef INFINI_CCL_BASE_GATHER_H_
#define INFINI_CCL_BASE_GATHER_H_

#include "communicator.h"
#include "data_type_impl.h"
#include "logging.h"
#include "operation.h"
#include "return_status_impl.h"

namespace infini::ccl {

template <BackendType backend_type, Device::Type device_type>
struct GatherImpl;

class Gather : public Operation<Gather> {
 public:
  template <BackendType backend_type, Device::Type device_type>
  static ReturnStatus Execute(const void *send_buff, void *recv_buff,
                              size_t count, DataType datatype, int root,
                              void *comm_handle, void *stream) {
    if (!comm_handle) {
      LOG("Invalid communicator handle for `Gather`.");
      return ReturnStatus::kInvalidArgument;
    }

    auto *comm = static_cast<Communicator *>(comm_handle);
    if (HasInvalidArgs(send_buff, recv_buff, datatype, root, comm)) {
      return ReturnStatus::kInvalidArgument;
    }
    if (count == 0) {
      return ReturnStatus::kSuccess;
    }

    return GatherImpl<backend_type, device_type>::Apply(
        send_buff, recv_buff, count, datatype, root, comm, stream);
  }

 private:
  static bool HasInvalidArgs(const void *send_buff, void *recv_buff,
                             DataType datatype, int root, Communicator *comm) {
    if (datatype < DataType::kChar || datatype >= DataType::kNumTypes) {
      LOG("Invalid data type for `Gather`.");
      return true;
    }
    if (root < 0 || root >= comm->size()) {
      LOG("Invalid root rank for `Gather`.");
      return true;
    }
    if (!send_buff) {
      LOG("Invalid send buffer pointer for `Gather`.");
      return true;
    }
    if (comm->rank() == root && !recv_buff) {
      LOG("Invalid root receive buffer pointer for `Gather`.");
      return true;
    }
    return false;
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BASE_GATHER_H_
