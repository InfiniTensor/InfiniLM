#ifndef INFINI_CCL_RETURN_STATUS_IMPL_H_
#define INFINI_CCL_RETURN_STATUS_IMPL_H_

#include <cstdint>

#include "return_status.h"

namespace infini::ccl {

enum class ReturnStatus : int8_t {
  kSuccess = infinicclSuccess,
  kUnhandledError = infinicclUnhandledError,
  kSystemError = infinicclSystemError,
  kInternalError = infinicclInternalError,
  kInvalidArgument = infinicclInvalidArgument,
  kInvalidUsage = infinicclInvalidUsage,
  kRemoteError = infinicclRemoteError,
  kInProgress = infinicclInProgress,
  kNotSupported = infinicclNotSupported,
  kNumResults = infinicclNumResults,
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_RETURN_STATUS_IMPL_H_
