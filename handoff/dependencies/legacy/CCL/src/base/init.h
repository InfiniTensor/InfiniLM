#ifndef INFINI_CCL_BASE_INIT_H_
#define INFINI_CCL_BASE_INIT_H_

#include "operation.h"
#include "return_status_impl.h"

namespace infini::ccl {

template <BackendType backend_type, Device::Type device_type>
struct InitImpl;

class Init : public Operation<Init> {
 public:
  template <BackendType backend_type, Device::Type device_type,
            typename... Args>
  static ReturnStatus Execute(Args &&...args) {
    return InitImpl<backend_type, device_type>::Apply(
        std::forward<Args>(args)...);
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BASE_INIT_H_
