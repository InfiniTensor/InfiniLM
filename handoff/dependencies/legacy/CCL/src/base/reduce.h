#ifndef INFINI_CCL_BASE_REDUCE_H_
#define INFINI_CCL_BASE_REDUCE_H_

#include "comm_impl.h"
#include "communicator.h"
#include "data_type_impl.h"
#include "logging.h"
#include "operation.h"
#include "return_status_impl.h"

namespace infini::ccl {

template <BackendType backend_type, Device::Type device_type>
struct ReduceImpl;

class Reduce : public Operation<Reduce> {
 public:
  template <BackendType backend_type, Device::Type device_type>
  static ReturnStatus Execute(const void *send_buff, void *recv_buff,
                              size_t count, DataType datatype,
                              ReductionOpType op, int root, void *comm_handle,
                              void *stream) {
    if (!comm_handle) {
      LOG("Invalid communicator handle for `Reduce`.");
      return ReturnStatus::kInvalidArgument;
    }

    auto *comm = static_cast<Communicator *>(comm_handle);
    if (HasInvalidArgs(send_buff, recv_buff, datatype, op, root, comm)) {
      return ReturnStatus::kInvalidArgument;
    }
    if (count == 0) {
      return ReturnStatus::kSuccess;
    }

    return ReduceImpl<backend_type, device_type>::Apply(
        send_buff, recv_buff, count, datatype, op, root, comm, stream);
  }

 private:
  static bool HasInvalidArgs(const void *send_buff, void *recv_buff,
                             DataType datatype, ReductionOpType op, int root,
                             Communicator *comm) {
    if (datatype < DataType::kChar || datatype >= DataType::kNumTypes) {
      LOG("Invalid data type for `Reduce`.");
      return true;
    }
    if (op < ReductionOpType::kSum || op >= ReductionOpType::kNumRedOps) {
      LOG("Invalid reduction operation for `Reduce`.");
      return true;
    }
    if (root < 0 || root >= comm->size()) {
      LOG("Invalid root rank for `Reduce`.");
      return true;
    }
    if (!send_buff) {
      LOG("Invalid send buffer pointer for `Reduce`.");
      return true;
    }
    if (comm->rank() == root && !recv_buff) {
      LOG("Invalid root receive buffer pointer for `Reduce`.");
      return true;
    }
    return false;
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BASE_REDUCE_H_
