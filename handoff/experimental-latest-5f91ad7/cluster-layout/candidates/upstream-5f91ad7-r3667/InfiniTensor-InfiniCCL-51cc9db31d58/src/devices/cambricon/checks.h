#ifndef INFINI_CCL_DEVICES_CAMBRICON_CHECKS_H_
#define INFINI_CCL_DEVICES_CAMBRICON_CHECKS_H_

#include <cnrt.h>

#include <iostream>

#include "return_status_impl.h"

#define INFINI_CHECK_CNRT(result) \
  ::infini::ccl::detail::CheckCnrtImpl((result), __FILE__, __LINE__)

namespace infini::ccl {

namespace detail {

inline ReturnStatus CheckCnrtImpl(cnrtRet_t cnrt_result, const char *file,
                                  int line) {
  if (cnrt_result != cnrtSuccess) {
    cnrtGetLastError();
    std::cerr << "CNRT error code: " << cnrt_result << " at line " << line
              << " in " << file << std::endl;
    std::abort();
  }
  return ReturnStatus::kSuccess;
}

}  // namespace detail

}  // namespace infini::ccl

#endif  // INFINI_CCL_DEVICES_CAMBRICON_CHECKS_H_
