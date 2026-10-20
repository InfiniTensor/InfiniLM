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
    if (!comm_handle) {
      LOG("Invalid communicator handle for `AllGather`.");
      return ReturnStatus::kInvalidArgument;
    }

    auto *comm = static_cast<Communicator *>(comm_handle);
    if (HasInvalidArgs(send_buff, recv_buff, count, datatype)) {
      return ReturnStatus::kInvalidArgument;
    }
    if (count == 0) {
      return ReturnStatus::kSuccess;
    }

    if (!comm->HasBackend(backend_type) || comm->device_type() != device_type) {
      using DispatchKey =
          typename BackendDependentType<backend_type, AllGather>::type;
      const BackendType comm_backend =
          Operation<DispatchKey>::FindSupportedBackend(
              comm->device_type(),
              {comm->HasBackend(backend_type) ? backend_type
                                              : BackendType::kCount,
               comm->intra_comm_backend(), comm->inter_comm_backend()});
      if (comm_backend == BackendType::kCount) {
        if (comm->intra_comm_backend() == BackendType::kCount &&
            comm->inter_comm_backend() == BackendType::kCount) {
          LOG("No initialized backend is available for `AllGather`.");
          return ReturnStatus::kInternalError;
        }
        return ReturnStatus::kNotSupported;
      }

      return Operation<DispatchKey>::Call(comm_backend, comm->device_type(),
                                          send_buff, recv_buff, count, datatype,
                                          comm_handle, stream);
    }

    return AllGatherImpl<backend_type, device_type>::Apply(
        send_buff, recv_buff, count, datatype, comm, stream);
  }

 private:
  template <BackendType, typename T>
  struct BackendDependentType {
    using type = T;
  };

  static bool HasInvalidArgs(const void *send_buff, void *recv_buff,
                             size_t count, DataType datatype) {
    if (datatype < DataType::kChar || datatype >= DataType::kNumTypes) {
      LOG("Invalid data type for `AllGather`.");
      return true;
    }
    if (count == 0) {
      return false;
    }
    if (!send_buff || !recv_buff) {
      LOG("Invalid buffer pointer for `AllGather`.");
      return true;
    }
    return false;
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BASE_ALL_GATHER_H_
