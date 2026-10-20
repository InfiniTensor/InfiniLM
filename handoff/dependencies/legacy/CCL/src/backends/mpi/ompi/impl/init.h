#ifndef INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_INIT_H_
#define INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_INIT_H_

#include "backends/mpi/ompi/checks.h"
#include "base/init.h"

namespace infini::ccl {

template <Device::Type device_type>
class InitImpl<BackendType::kOmpi, device_type> {
 public:
  static ReturnStatus Apply(int *argc, char ***argv) {
    int initialized = 0;
    INFINI_CHECK_MPI(MPI_Initialized(&initialized));

    if (!initialized) {
      int provided;
      // Use `MPI_Init_thread` to support multi-threaded GPU streams.
      INFINI_CHECK_MPI(MPI_Init_thread(argc, argv, required_level_, &provided));

      if (provided < required_level_) {
        // TODO(lzm): change to use `glog`.
        std::cerr
            << "[InfiniCCL Warning] MPI implementation does not fully support "
            << "`MPI_THREAD_SERIALIZED` (provided: " << provided << "). "
            << "Concurrent collective operations may be unsafe." << std::endl;
      }
    }
    return ReturnStatus::kSuccess;
  }

 private:
  constexpr static auto required_level_ = MPI_THREAD_FUNNELED;
};

template <>
struct BackendEnabled<Init, BackendType::kOmpi> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_INIT_H_
