#ifndef INFINI_CCL_BACKENDS_MPI_OMPI_COMM_INSTANCE_H_
#define INFINI_CCL_BACKENDS_MPI_OMPI_COMM_INSTANCE_H_

#include <mpi.h>

#include "checks.h"
#include "communicator.h"

namespace infini::ccl {

struct OmpiInstance : public BackendCommInstance {
  MPI_Comm handle = MPI_COMM_NULL;

  OmpiInstance() { type = BackendType::kOmpi; }
  ~OmpiInstance() override { Destroy(); }

  void Destroy() {
    if (handle != MPI_COMM_WORLD && handle != MPI_COMM_NULL) {
      INFINI_CHECK_MPI(MPI_Comm_free(&handle));
      handle = MPI_COMM_NULL;
    }
    return;
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_MPI_OMPI_COMM_INSTANCE_H_
