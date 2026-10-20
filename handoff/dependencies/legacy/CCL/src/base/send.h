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
  static ReturnStatus Execute(const void* send_buff, size_t count,
                              DataType datatype, int peer, void* comm_handle,
                              void* stream) {
    if (!comm_handle) {
      LOG("Invalid communicator handle for `Send`.");
      return ReturnStatus::kInvalidArgument;
    }

    auto* comm = static_cast<Communicator*>(comm_handle);
    if (HasInvalidArgs(send_buff, count, datatype, peer, comm)) {
      return ReturnStatus::kInvalidArgument;
    }
    if (count == 0) {
      return ReturnStatus::kSuccess;
    }

    if (!comm->HasBackend(backend_type) || comm->device_type() != device_type) {
      using DispatchKey =
          typename BackendDependentType<backend_type, Send>::type;
      const BackendType comm_backend =
          Operation<DispatchKey>::FindSupportedBackend(
              comm->device_type(),
              {comm->HasBackend(backend_type) ? backend_type
                                              : BackendType::kCount,
               comm->intra_comm_backend(), comm->inter_comm_backend()});
      if (comm_backend == BackendType::kCount) {
        if (comm->intra_comm_backend() == BackendType::kCount &&
            comm->inter_comm_backend() == BackendType::kCount) {
          LOG("No initialized backend is available for `Send`.");
          return ReturnStatus::kInternalError;
        }
        return ReturnStatus::kNotSupported;
      }

      return Operation<DispatchKey>::Call(comm_backend, comm->device_type(),
                                          send_buff, count, datatype, peer,
                                          comm_handle, stream);
    }

    return SendImpl<backend_type, device_type>::Apply(
        send_buff, count, datatype, peer, comm, stream);
  }

 private:
  template <BackendType, typename T>
  struct BackendDependentType {
    using type = T;
  };

  static bool HasInvalidArgs(const void* send_buff, size_t count,
                             DataType datatype, int peer, Communicator* comm) {
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
