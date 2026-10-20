#ifndef INFINI_CCL_BACKENDS_CCL_NCCL_CHECKS_H_
#define INFINI_CCL_BACKENDS_CCL_NCCL_CHECKS_H_

#include <iostream>

#include "backends/ccl/nccl/compat.h"
#include "return_status_impl.h"

#define INFINI_CHECK_NCCL(result) \
  ::infini::ccl::detail::CheckNcclImpl((result), __FILE__, __LINE__)

namespace infini::ccl {

namespace detail {

inline ReturnStatus CheckNcclImpl(ncclResult_t nccl_result, const char *file,
                                  int line) {
  if (nccl_result != ncclSuccess) {
    std::cerr << "backend(nccl) NCCL error code: " << nccl_result << " at line "
              << line << " in " << file << std::endl;
    std::abort();
  }
  return ReturnStatus::kSuccess;
}

}  // namespace detail

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_NCCL_CHECKS_H_
