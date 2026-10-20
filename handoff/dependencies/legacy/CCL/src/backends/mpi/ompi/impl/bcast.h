#ifndef INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_BCAST_H_
#define INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_BCAST_H_

#include "base/bcast.h"

namespace infini::ccl {

template <>
struct BackendEnabled<Bcast, BackendType::kOmpi> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_BCAST_H_
