#ifndef INFINI_CCL_BACKENDS_MPI_OMPI_CHECKS_H_
#define INFINI_CCL_BACKENDS_MPI_OMPI_CHECKS_H_

#include <mpi.h>

#include <iostream>

#include "return_status_impl.h"

#define INFINI_CHECK_MPI(result) \
  ::infini::ccl::detail::CheckMpiImpl((result), __FILE__, __LINE__)

namespace infini::ccl {

namespace detail {

inline ReturnStatus CheckMpiImpl(int mpi_result, const char *file, int line) {
  if (mpi_result != MPI_SUCCESS) {
    std::cerr << "backend(ompi) MPI error code: " << mpi_result << " at line "
              << line << " in " << file << std::endl;
    std::abort();
  }
  return ReturnStatus::kSuccess;
}

}  // namespace detail

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_MPI_OMPI_CHECKS_H_
