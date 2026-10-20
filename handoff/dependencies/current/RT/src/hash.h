#ifndef INFINI_RT_HASH_H_
#define INFINI_RT_HASH_H_

#include <functional>
#include <optional>
#include <vector>

template <typename T>
inline void HashCombine(std::size_t& seed, const T& v) {
  std::hash<std::decay_t<decltype(v)>> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <typename T>
inline void HashCombine(std::size_t& seed, const std::vector<T>& v) {
  HashCombine(seed, v.size());
  for (const auto& elem : v) {
    HashCombine(seed, elem);
  }
}

template <typename T>
inline void HashCombine(std::size_t& seed, const std::optional<T>& v) {
  HashCombine(seed, v.has_value());
  if (v.has_value()) {
    HashCombine(seed, *v);
  }
}

#endif
