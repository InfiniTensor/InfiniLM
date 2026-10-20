#ifndef INFINI_OPS_COMMON_OP_UTILS_REDUCTION_H_
#define INFINI_OPS_COMMON_OP_UTILS_REDUCTION_H_

#include <cassert>
#include <cstdint>
#include <optional>
#include <string>

namespace infini::ops::reduction_detail {

inline int64_t FromString(const std::string& reduction) {
  if (reduction == "none") {
    return 0;
  }

  if (reduction == "mean") {
    return 1;
  }

  assert(reduction == "sum" && "`reduction` must be `none`, `mean`, or `sum`");

  return 2;
}

inline int64_t FromPythonArguments(const std::optional<bool> size_average,
                                   const std::optional<bool> reduce,
                                   const std::string& reduction) {
  if (!size_average.has_value() && !reduce.has_value()) {
    return FromString(reduction);
  }

  if (!reduce.value_or(true)) {
    return 0;
  }

  if (!size_average.value_or(true)) {
    return 2;
  }

  return 1;
}

}  // namespace infini::ops::reduction_detail

#endif
