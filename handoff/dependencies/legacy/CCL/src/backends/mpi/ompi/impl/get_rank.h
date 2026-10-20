#ifndef INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_GET_RANK_H_
#define INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_GET_RANK_H_

#include "backends/mpi/ompi/checks.h"
#include "base/get_rank.h"

namespace infini::ccl {

template <Device::Type device_type>
class GetRankImpl<BackendType::kOmpi, device_type> {
 public:
  static ReturnStatus Apply(int *rank) {
    INFINI_CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, rank));
    return ReturnStatus::kSuccess;
  }
};

template <>
struct BackendEnabled<GetRank, BackendType::kOmpi> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_GET_RANK_H_
