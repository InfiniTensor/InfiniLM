#ifndef INFINI_CCL_BASE_COMM_INIT_RANK_H_
#define INFINI_CCL_BASE_COMM_INIT_RANK_H_

#include "logging.h"
#include "operation.h"
#include "return_status_impl.h"

namespace infini::ccl {

template <BackendType backend_type, Device::Type device_type>
struct CommInitRankImpl;

class CommInitRank : public Operation<CommInitRank> {
 public:
  template <BackendType backend_type, Device::Type device_type,
            typename... Args>
  static ReturnStatus Execute(void **comm_handle, Args &&...args) {
    Communicator *&comm = *reinterpret_cast<Communicator **>(comm_handle);
    if (comm && comm->intra_comm()) {
      // TODO(lzm): change to use `glog`.
      LOG("Invalid communicator handle for `CommInitRank`.");
      return ReturnStatus::kInvalidArgument;
    }

    constexpr Device::Type kDev =
        ListGetBest<DevicePriority>(ActiveDevices<CommInitRank>{});
    using Rt = Runtime<kDev>;

    int current_dev = 0;
    CHECK_STATUS(Rt, Rt::GetDevice(&current_dev));

    if (!comm) {
      comm = new Communicator(kDev, current_dev);
    } else {
      comm->set_device_id(current_dev);
    }

    return CommInitRankImpl<backend_type, device_type>::Apply(
        comm, std::forward<Args>(args)...);
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BASE_COMM_INIT_RANK_H_
