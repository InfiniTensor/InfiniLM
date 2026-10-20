#ifndef INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_GET_SIZE_H_
#define INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_GET_SIZE_H_

#include "backends/mpi/ompi/checks.h"
#include "base/get_size.h"

namespace infini::ccl {

template <Device::Type device_type>
class GetSizeImpl<BackendType::kOmpi, device_type> {
 public:
  static ReturnStatus Apply(int *size) {
    INFINI_CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, size));
    return ReturnStatus::kSuccess;
  }
};

template <>
struct BackendEnabled<GetSize, BackendType::kOmpi> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_GET_SIZE_H_
