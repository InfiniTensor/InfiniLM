#ifndef INFINI_CCL_BACKENDS_CCL_MCCL_CHECKS_H_
#define INFINI_CCL_BACKENDS_CCL_MCCL_CHECKS_H_

#include <mccl.h>

#include <iostream>

#include "return_status_impl.h"

#define INFINI_CHECK_MCCL(result) \
  ::infini::ccl::detail::CheckMcclImpl((result), __FILE__, __LINE__)

namespace infini::ccl {

namespace detail {

inline ReturnStatus CheckMcclImpl(mcclResult_t mccl_result, const char *file,
                                  int line) {
  if (mccl_result != mcclSuccess) {
    std::cerr << "backend(mccl) MCCL error code: " << mccl_result << " at line "
              << line << " in " << file << std::endl;
    std::abort();
  }
  return ReturnStatus::kSuccess;
}

}  // namespace detail

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_MCCL_CHECKS_H_
