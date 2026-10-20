#ifndef INFINI_OPS_COMMON_GENERIC_UTILS_H_
#define INFINI_OPS_COMMON_GENERIC_UTILS_H_

#include <cstddef>

namespace infini::ops::utils {

inline std::size_t IndexToOffset(std::size_t flat_index, std::size_t ndim,
                                 const std::size_t* shape,
                                 const std::ptrdiff_t* strides) {
  std::size_t res = 0;
  for (std::size_t i = ndim; i-- > 0;) {
    res += (flat_index % shape[i]) * strides[i];
    flat_index /= shape[i];
  }
  return res;
}

template <typename X, typename Y>
constexpr auto CeilDiv(const X& x, const Y& y) {
  return (x + y - 1) / y;
}

}  // namespace infini::ops::utils

#endif
