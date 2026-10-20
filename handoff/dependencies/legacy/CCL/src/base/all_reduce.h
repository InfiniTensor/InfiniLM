#ifndef INFINI_CCL_BASE_ALL_REDUCE_H_
#define INFINI_CCL_BASE_ALL_REDUCE_H_

#include "comm_impl.h"
#include "communicator.h"
#include "logging.h"
#include "operation.h"
#include "return_status_impl.h"

namespace infini::ccl {

template <BackendType backend_type, Device::Type device_type>
struct AllReduceImpl;

struct AllReduceCaptureQuery {
  void *comm_handle;
  int *supported;
};

template <BackendType backend_type, Device::Type device_type>
struct AllReduceCaptureImpl {
  static ReturnStatus Query(Communicator *, int *supported) {
    *supported = 0;
    return ReturnStatus::kSuccess;
  }
};

class AllReduce : public Operation<AllReduce> {
 public:
  template <typename Key = AllReduce>
  static ReturnStatus QueryCapture(const AllReduceCaptureQuery &query) {
    if constexpr (ListSize<ActiveBackends<Key>>::value == 0 ||
                  ListSize<ActiveDevices<Key>>::value == 0) {
      return ReturnStatus::kNotSupported;
    } else {
      return Operation<Key>::Call(query);
    }
  }

  template <BackendType backend_type, Device::Type device_type>
  static ReturnStatus Execute(const AllReduceCaptureQuery &query) {
    if (!query.supported) {
      return ReturnStatus::kInvalidArgument;
    }
    *query.supported = 0;
    if (!query.comm_handle) {
      return ReturnStatus::kInvalidArgument;
    }
    return AllReduceCaptureImpl<backend_type, device_type>::Query(
        static_cast<Communicator *>(query.comm_handle), query.supported);
  }

  template <BackendType backend_type, Device::Type device_type,
            typename... Args>
  static ReturnStatus Execute(const void *send_buff, void *recv_buff,
                              size_t count, DataType datatype,
                              ReductionOpType op, void *comm_handle,
                              void *stream) {
    if (HasInvalidArgs(send_buff, recv_buff, datatype, op, comm_handle)) {
      return ReturnStatus::kInvalidArgument;
    }
    auto *comm = static_cast<Communicator *>(comm_handle);
    return AllReduceImpl<backend_type, device_type>::Apply(
        send_buff, recv_buff, count, datatype, op, comm, stream);
  }

 private:
  static bool HasInvalidArgs(const void *send_buff, void *recv_buff,
                             DataType datatype, ReductionOpType op,
                             void *comm_handle) {
    if (!comm_handle) {
      // TODO(lzm): change to use `glog`.
      LOG("Invalid communicator handle for AllReduce.");
      return true;
    }
    if (!send_buff || !recv_buff) {
      LOG("Invalid buffer pointer for AllReduce.");
      return true;
    }
    if (op < ReductionOpType::kSum || op >= ReductionOpType::kNumRedOps) {
      LOG("Invalid reduction operation for AllReduce.");
      return true;
    }
    if (datatype < DataType::kChar || datatype >= DataType::kNumTypes) {
      LOG("Invalid data type for AllReduce.");
      return true;
    }
    return false;
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BASE_ALL_REDUCE_H_
