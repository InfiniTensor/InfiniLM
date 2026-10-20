#ifndef INFINI_CCL_BACKENDS_CCL_COMMON_IMPL_ALL_REDUCE_H_
#define INFINI_CCL_BACKENDS_CCL_COMMON_IMPL_ALL_REDUCE_H_

#include "backends/ccl/common/api.h"
#include "backends/ccl/common/comm_instance.h"
#include "base/all_reduce.h"
#include "communicator.h"

namespace infini::ccl {

template <BackendType backend, Device::Type device>
class CclAllReduceImpl {
 public:
  static ReturnStatus Apply(const void *send_buff, void *recv_buff,
                            size_t count, DataType data_type,
                            ReductionOpType op, Communicator *comm,
                            void *stream) {
    using Api = CclApi<backend, device>;
    using TypeMap = CclTypeMap<backend, device>;
    using CommInstance = CclCommInstance<Api>;

    auto *comm_internal = static_cast<Communicator *>(comm);
    if (!comm_internal) {
      return ReturnStatus::kInternalError;
    }

    auto *intra = static_cast<CommInstance *>(comm_internal->intra_comm());
    if (!intra || !intra->handle) {
      return ReturnStatus::kInternalError;
    }

    typename Api::DataType ccl_type{};
    if (!TypeMap::ToBackendDataType(data_type, &ccl_type)) {
      return ReturnStatus::kNotSupported;
    }

    typename Api::RedOp ccl_op{};
    if (!TypeMap::ToBackendRedOp(op, &ccl_op)) {
      return ReturnStatus::kNotSupported;
    }

    return Api::Check(Api::AllReduce(
        send_buff, recv_buff, count, ccl_type, ccl_op, intra->handle,
        reinterpret_cast<typename Api::Stream>(stream)));
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_COMMON_IMPL_ALL_REDUCE_H_
