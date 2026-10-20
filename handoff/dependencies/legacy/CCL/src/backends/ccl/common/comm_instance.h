#ifndef INFINI_CCL_BACKENDS_CCL_COMMON_COMM_INSTANCE_H_
#define INFINI_CCL_BACKENDS_CCL_COMMON_COMM_INSTANCE_H_

#include "backends/ccl/common/api.h"
#include "communicator.h"

namespace infini::ccl {

template <typename Api>
struct CclCommInstance : public BackendCommInstance {
  using Comm = typename Api::Comm;

  Comm handle{};

  CclCommInstance() { type = Api::kBackendType; }
  ~CclCommInstance() override { Destroy(); }

  void Destroy() {
    if (handle != Comm{}) {
      (void)Api::Check(Api::CommDestroy(handle));
      handle = Comm{};
    }
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_COMMON_COMM_INSTANCE_H_
