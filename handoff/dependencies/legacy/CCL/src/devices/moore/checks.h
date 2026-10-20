#ifndef INFINI_CCL_DEVICES_MOORE_CHECKS_H_
#define INFINI_CCL_DEVICES_MOORE_CHECKS_H_

#include <musa_runtime.h>

#include <iostream>

#include "return_status_impl.h"

#define INFINI_CHECK_MUSA(result) \
  ::infini::ccl::detail::CheckMusaImpl((result), __FILE__, __LINE__)

namespace infini::ccl {

namespace detail {

inline ReturnStatus CheckMusaImpl(musaError_t musa_result, const char *file,
                                  int line) {
  if (musa_result != musaSuccess) {
    musaGetLastError();
    std::cerr << "MUSA error code: " << musa_result << " at line " << line
              << " in " << file << std::endl;
    std::abort();
  }
  return ReturnStatus::kSuccess;
}

}  // namespace detail

}  // namespace infini::ccl

#endif  // INFINI_CCL_DEVICES_MOORE_CHECKS_H_
