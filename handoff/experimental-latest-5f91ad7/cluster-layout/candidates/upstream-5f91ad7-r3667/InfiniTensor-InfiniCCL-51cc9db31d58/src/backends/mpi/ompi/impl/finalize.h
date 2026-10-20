#ifndef INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_FINALIZE_H_
#define INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_FINALIZE_H_

#include "backends/mpi/ompi/checks.h"
#include "base/finalize.h"

namespace infini::ccl {

template <Device::Type device_type>
class FinalizeImpl<BackendType::kOmpi, device_type> {
 public:
  static ReturnStatus Apply() {
    int finalized = 0;
    INFINI_CHECK_MPI(MPI_Finalized(&finalized));

    if (!finalized) {
      INFINI_CHECK_MPI(MPI_Finalize());
    }

    return ReturnStatus::kSuccess;
  }
};

template <>
struct BackendEnabled<Finalize, BackendType::kOmpi> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_FINALIZE_H_
