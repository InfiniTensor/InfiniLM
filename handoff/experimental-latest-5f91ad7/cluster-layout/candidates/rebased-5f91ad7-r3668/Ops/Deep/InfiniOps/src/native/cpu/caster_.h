#ifndef INFINI_OPS_COMMON_CPU_CASTER_H_
#define INFINI_OPS_COMMON_CPU_CASTER_H_

#include <infini/rt/cpu/data_type_.h>

#include <type_traits>

#include "caster.h"

namespace infini::ops {

template <>
struct Caster<Device::Type::kCpu> {
  template <typename Dst, typename Src>
  static Dst Cast(Src&& x) {
    static_assert(!std::is_reference_v<Dst>,
                  "`Cast` cannot return reference types");

    using PureDst = std::remove_cv_t<std::remove_reference_t<Dst>>;
    using PureSrc = std::remove_cv_t<std::remove_reference_t<Src>>;

    if constexpr (std::is_same_v<PureDst, PureSrc>) {
      return std::forward<Src>(x);
    }

    constexpr bool src_is_custom = IsBFloat16<Device::Type::kCpu, PureSrc> ||
                                   IsFP16<Device::Type::kCpu, PureSrc>;
    constexpr bool dst_is_custom = IsBFloat16<Device::Type::kCpu, PureDst> ||
                                   IsFP16<Device::Type::kCpu, PureDst>;

    if constexpr (!src_is_custom && !dst_is_custom) {
      return static_cast<PureDst>(std::forward<Src>(x));
    } else {
      return FromFloatHelper<PureDst>(ToFloatHelper(std::forward<Src>(x)));
    }
  }

 private:
  template <typename T, typename = void>
  struct HasToFloat : std::false_type {};

  template <typename T>
  struct HasToFloat<T, std::void_t<decltype(std::declval<T>().ToFloat())>>
      : std::true_type {};

  template <typename T, typename = void>
  struct HasFromFloat : std::false_type {};

  template <typename T>
  struct HasFromFloat<
      T, std::void_t<decltype(T::FromFloat(std::declval<float>()))>>
      : std::true_type {};

  template <typename T>
  static constexpr float ToFloatHelper(T&& x) {
    if constexpr (HasToFloat<T>::value) {
      return std::forward<T>(x).ToFloat();
    } else {
      return static_cast<float>(x);
    }
  }

  template <typename PureDst>
  static constexpr PureDst FromFloatHelper(float f) {
    if constexpr (HasFromFloat<PureDst>::value) {
      return PureDst::FromFloat(f);
    } else {
      return static_cast<PureDst>(f);
    }
  }
};

}  // namespace infini::ops

#endif
