#ifndef INFINI_CCL_BASE_FINALIZE_H_
#define INFINI_CCL_BASE_FINALIZE_H_

#include "operation.h"
#include "return_status_impl.h"

namespace infini::ccl {

template <BackendType backend_type, Device::Type device_type>
struct FinalizeImpl;

class Finalize : public Operation<Finalize> {
 public:
  template <BackendType backend_type, Device::Type device_type,
            typename... Args>
  static ReturnStatus Execute(Args &&...args) {
    return FinalizeImpl<backend_type, device_type>::Apply(
        std::forward<Args>(args)...);
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BASE_FINALIZE_H_
