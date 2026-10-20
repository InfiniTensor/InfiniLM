#ifndef INFINI_CCL_BACKENDS_CCL_COMMON_IMPL_RECV_H_
#define INFINI_CCL_BACKENDS_CCL_COMMON_IMPL_RECV_H_

#include "backends/ccl/common/api.h"
#include "backends/ccl/common/comm_instance.h"
#include "base/recv.h"
#include "communicator.h"

namespace infini::ccl {

template <BackendType backend, Device::Type device>
class CclRecvImpl {
 public:
  static ReturnStatus Apply(void* recv_buff, size_t count, DataType data_type,
                            int peer, Communicator* comm, void* stream) {
    using Api = CclApi<backend, device>;
    using TypeMap = CclTypeMap<backend, device>;
    using CommInstance = CclCommInstance<Api>;

    if (!comm || !comm->intra_comm() || comm->intra_comm_backend() != backend ||
        comm->device_type() != device) {
      return ReturnStatus::kInternalError;
    }

    auto* intra = static_cast<CommInstance*>(comm->intra_comm());
    if (!intra->handle) {
      return ReturnStatus::kInternalError;
    }

    typename Api::DataType ccl_type{};
    if (!TypeMap::ToBackendDataType(data_type, &ccl_type)) {
      return ReturnStatus::kNotSupported;
    }

    return Api::Check(
        Api::Recv(recv_buff, count, ccl_type, peer, intra->handle,
                  reinterpret_cast<typename Api::Stream>(stream)));
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_COMMON_IMPL_RECV_H_
