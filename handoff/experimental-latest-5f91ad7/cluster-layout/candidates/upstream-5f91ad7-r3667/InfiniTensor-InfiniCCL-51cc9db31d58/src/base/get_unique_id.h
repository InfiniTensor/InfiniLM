#ifndef INFINI_CCL_BASE_GET_UNIQUE_ID_H_
#define INFINI_CCL_BASE_GET_UNIQUE_ID_H_

#include "comm.h"
#include "operation.h"
#include "return_status_impl.h"

namespace infini::ccl {

template <BackendType backend_type, Device::Type device_type>
struct GetUniqueIdImpl;

class GetUniqueId : public Operation<GetUniqueId> {
 public:
  template <BackendType backend_type, Device::Type device_type>
  static ReturnStatus Execute(infinicclUniqueId *id) {
    return GetUniqueIdImpl<backend_type, device_type>::Apply(id);
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BASE_GET_UNIQUE_ID_H_
