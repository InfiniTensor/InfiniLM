#ifndef INFINI_CCL_BASE_COMM_INIT_ALL_H_
#define INFINI_CCL_BASE_COMM_INIT_ALL_H_

#include "communicator.h"
#include "logging.h"
#include "operation.h"
#include "return_status_impl.h"

namespace infini::ccl {

template <BackendType backend_type, Device::Type device_type>
struct CommInitAllImpl;

class CommInitAll : public Operation<CommInitAll> {
 public:
  template <BackendType backend_type, Device::Type device_type,
            typename... Args>
  static ReturnStatus Execute(void **comm_handle, Args &&...args) {
    Communicator *&comm = *reinterpret_cast<Communicator **>(comm_handle);
    if (comm && comm->inter_comm()) {
      // TODO(lzm): change to use `glog`.
      LOG("Invalid communicator handle for `CommInitAll`.");
      return ReturnStatus::kInvalidArgument;
    }

    constexpr Device::Type kDev =
        ListGetBest<DevicePriority>(ActiveDevices<CommInitAll>{});

    if (!comm) {
      comm = new Communicator(kDev, 0);
    }

    return CommInitAllImpl<backend_type, device_type>::Apply(
        comm, std::forward<Args>(args)...);
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BASE_COMM_INIT_ALL_H_
