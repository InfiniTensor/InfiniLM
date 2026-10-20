#ifndef INFINI_CCL_COMM_IMPL_H_
#define INFINI_CCL_COMM_IMPL_H_

#include <cstdint>

#include "comm.h"

namespace infini::ccl {

enum class ReductionOpType : int8_t {
  kSum = infinicclSum,
  kProd = infinicclProd,
  kMax = infinicclMax,
  kMin = infinicclMin,
  kAvg = infinicclAvg,
  kNumRedOps = infinicclNumOps,
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_COMM_IMPL_H_
