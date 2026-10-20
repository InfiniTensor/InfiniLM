#ifndef INFINI_CCL_BASE_COMM_DESTROY_H_
#define INFINI_CCL_BASE_COMM_DESTROY_H_

#include "communicator.h"
#include "logging.h"
#include "operation.h"
#include "return_status_impl.h"

namespace infini::ccl {

template <BackendType backend_type, Device::Type device_type>
struct CommDestroyImpl;

class CommDestroy : public Operation<CommDestroy> {
 public:
  template <BackendType backend_type, Device::Type device_type>
  static ReturnStatus Execute(void *comm_handle) {
    if (!comm_handle) {
      // TODO(lzm): change to use `glog`.
      LOG("Invalid communicator handle for `CommDestroy`.");
      return ReturnStatus::kInvalidArgument;
    }

    auto status =
        CommDestroyImpl<backend_type, device_type>::Apply(comm_handle);

    if (status == ReturnStatus::kSuccess) {
      delete static_cast<Communicator *>(comm_handle);
    }

    return status;
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BASE_COMM_DESTROY_H_
