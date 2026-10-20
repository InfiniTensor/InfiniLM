#ifndef INFINI_RT_COMMON_METADATA_VIEW_H_
#define INFINI_RT_COMMON_METADATA_VIEW_H_

#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

namespace infini::rt::detail {

template <typename T, std::size_t inline_capacity>
class SmallVector;

template <typename T>
class MetadataView;

template <typename T>
struct IsMetadataView : std::false_type {};

template <typename T>
struct IsMetadataView<MetadataView<T>> : std::true_type {};

template <typename T>
struct IsMetadataViewSmallVector : std::false_type {};

template <typename T, std::size_t inline_capacity>
struct IsMetadataViewSmallVector<SmallVector<T, inline_capacity>>
    : std::true_type {};

template <typename Range, typename T, typename = void>
struct IsMetadataViewComparableRange : std::false_type {};

template <typename Range, typename T>
struct IsMetadataViewComparableRange<
    Range, T,
    std::void_t<decltype(std::begin(std::declval<const Range&>())),
                decltype(std::end(std::declval<const Range&>())),
                decltype(std::size(std::declval<const Range&>())),
                decltype(static_cast<bool>(
                    std::declval<const T&>() ==
                    *std::begin(std::declval<const Range&>())))>>
    : std::true_type {};

template <typename T>
class MetadataView {
 public:
  using value_type = T;

  using size_type = std::size_t;

  using reference = const T&;

  using const_reference = const T&;

  using pointer = const T*;

  using const_pointer = const T*;

  using iterator = const T*;

  using const_iterator = const T*;

  constexpr MetadataView() noexcept = default;

  constexpr MetadataView(const_pointer data, size_type size) noexcept
      : data_{data}, size_{size} {}

  constexpr size_type size() const noexcept { return size_; }

  constexpr bool empty() const noexcept { return size_ == 0; }

  constexpr const_pointer data() const noexcept { return data_; }

  constexpr const_reference front() const noexcept { return data_[0]; }

  constexpr const_reference back() const noexcept { return data_[size_ - 1]; }

  constexpr const_reference operator[](size_type index) const noexcept {
    return data_[index];
  }

  constexpr const_iterator begin() const noexcept { return data_; }

  constexpr const_iterator end() const noexcept {
    return empty() ? data_ : data_ + size_;
  }

  constexpr const_iterator cbegin() const noexcept { return begin(); }

  constexpr const_iterator cend() const noexcept { return end(); }

 private:
  const_pointer data_{nullptr};

  size_type size_{0};
};

template <typename Left, typename Right,
          std::enable_if_t<
              IsMetadataViewComparableRange<MetadataView<Right>, Left>::value,
              int> = 0>
constexpr bool operator==(MetadataView<Left> left, MetadataView<Right> right) {
  if (left.size() != right.size()) return false;

  for (std::size_t index = 0; index < left.size(); ++index) {
    if (!(left[index] == right[index])) return false;
  }

  return true;
}

template <typename Left, typename Right,
          std::enable_if_t<
              IsMetadataViewComparableRange<MetadataView<Right>, Left>::value,
              int> = 0>
constexpr bool operator!=(MetadataView<Left> left, MetadataView<Right> right) {
  return !(left == right);
}

template <typename T, typename Range,
          std::enable_if_t<
              !IsMetadataView<std::decay_t<Range>>::value &&
                  !IsMetadataViewSmallVector<std::decay_t<Range>>::value &&
                  IsMetadataViewComparableRange<Range, T>::value,
              int> = 0>
constexpr bool operator==(MetadataView<T> left, const Range& right) {
  if (left.size() != static_cast<std::size_t>(std::size(right))) return false;

  auto right_iterator = std::begin(right);
  for (std::size_t index = 0; index < left.size(); ++index, ++right_iterator) {
    if (!(left[index] == *right_iterator)) return false;
  }

  return true;
}

template <typename Range, typename T,
          std::enable_if_t<
              !IsMetadataView<std::decay_t<Range>>::value &&
                  !IsMetadataViewSmallVector<std::decay_t<Range>>::value &&
                  IsMetadataViewComparableRange<Range, T>::value,
              int> = 0>
constexpr bool operator==(const Range& left, MetadataView<T> right) {
  return right == left;
}

template <typename T, typename Range,
          std::enable_if_t<
              !IsMetadataView<std::decay_t<Range>>::value &&
                  !IsMetadataViewSmallVector<std::decay_t<Range>>::value &&
                  IsMetadataViewComparableRange<Range, T>::value,
              int> = 0>
constexpr bool operator!=(MetadataView<T> left, const Range& right) {
  return !(left == right);
}

template <typename Range, typename T,
          std::enable_if_t<
              !IsMetadataView<std::decay_t<Range>>::value &&
                  !IsMetadataViewSmallVector<std::decay_t<Range>>::value &&
                  IsMetadataViewComparableRange<Range, T>::value,
              int> = 0>
constexpr bool operator!=(const Range& left, MetadataView<T> right) {
  return !(right == left);
}

}  // namespace infini::rt::detail

#endif
