#ifndef INFINI_CCL_DEVICES_METAX_CHECKS_H_
#define INFINI_CCL_DEVICES_METAX_CHECKS_H_

// clang-format off
#include <mcr/mc_runtime.h>
// clang-format on

#include <iostream>

#include "return_status_impl.h"

#define INFINI_CHECK_MACA(result) \
  ::infini::ccl::detail::CheckMacaImpl((result), __FILE__, __LINE__)

namespace infini::ccl {

namespace detail {

inline ReturnStatus CheckMacaImpl(mcError_t maca_result, const char *file,
                                  int line) {
  if (maca_result != mcSuccess) {
    mcGetLastError();
    std::cerr << "MACA error code: " << maca_result << " at line " << line
              << " in " << file << std::endl;
    std::abort();
  }
  return ReturnStatus::kSuccess;
}

}  // namespace detail

}  // namespace infini::ccl

#endif  // INFINI_CCL_DEVICES_METAX_CHECKS_H_
