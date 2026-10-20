#ifndef INFINI_OPS_COMMON_CONSTEXPR_MAP_H_
#define INFINI_OPS_COMMON_CONSTEXPR_MAP_H_

#include <array>
#include <cassert>
#include <cstdlib>
#include <utility>

namespace infini::ops {

template <typename Key, typename Value, std::size_t size>
struct ConstexprMap {
  constexpr ConstexprMap(std::array<std::pair<Key, Value>, size> data)
      : data_(data) {}

  constexpr Value at(Key key) const {
    for (const auto& pr : data_) {
      if (pr.first == key) return pr.second;
    }
    // TODO(lzm): change to logging.
    assert("the key is not found in the `ConstexprMap`");
    // Unreachable, provided to satisfy the compiler's requirement.
    std::abort();
  }

 private:
  std::array<std::pair<Key, Value>, size> data_;
};

}  // namespace infini::ops

#endif
