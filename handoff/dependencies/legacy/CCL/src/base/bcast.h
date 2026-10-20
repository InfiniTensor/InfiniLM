#ifndef INFINI_CCL_BASE_BCAST_H_
#define INFINI_CCL_BASE_BCAST_H_

#include "base/broadcast.h"
#include "operation.h"
#include "return_status_impl.h"

namespace infini::ccl {

class Bcast : public Operation<Bcast> {
 public:
  template <BackendType backend_type, Device::Type device_type>
  static ReturnStatus Execute(void *buff, size_t count, DataType datatype,
                              int root, void *comm_handle, void *stream) {
    // Note: we simply forward to `Broadcast`.
    return Broadcast::Execute<backend_type, device_type>(
        buff, buff, count, datatype, root, comm_handle, stream);
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BASE_BCAST_H_
