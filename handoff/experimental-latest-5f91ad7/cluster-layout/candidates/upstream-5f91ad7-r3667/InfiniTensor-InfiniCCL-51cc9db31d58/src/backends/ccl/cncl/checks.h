#ifndef INFINI_CCL_BACKENDS_CCL_CNCL_CHECKS_H_
#define INFINI_CCL_BACKENDS_CCL_CNCL_CHECKS_H_

#include <cncl.h>

#include <iostream>

#include "return_status_impl.h"

#define INFINI_CHECK_CNCL(result) \
  ::infini::ccl::detail::CheckCnclImpl((result), __FILE__, __LINE__)

namespace infini::ccl {

namespace detail {

inline ReturnStatus CheckCnclImpl(cnclResult_t cncl_result, const char *file,
                                  int line) {
  if (cncl_result != CNCL_RET_SUCCESS) {
    std::cerr << "backend(cncl) CNCL error code: " << cncl_result << " at line "
              << line << " in " << file << std::endl;
    std::abort();
  }
  return ReturnStatus::kSuccess;
}

}  // namespace detail

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_CNCL_CHECKS_H_
