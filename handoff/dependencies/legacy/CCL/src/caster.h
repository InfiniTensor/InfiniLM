#ifndef INFINI_CCL_CASTER_H_
#define INFINI_CCL_CASTER_H_

#include <type_traits>
#include <typeinfo>
#include <utility>

#include "device.h"

namespace infini::ccl {

// Check if a type is complete (i.e. fully defined) at compile-time.
template <typename T, typename = void>
struct IsComplete : std::false_type {};

template <typename T>
struct IsComplete<T, std::void_t<decltype(sizeof(T))>> : std::true_type {};

template <Device::Type kDev, typename Dst, typename Src>
struct HardwareCastImpl;

template <Device::Type kDev, typename Dst, typename Src>
inline constexpr bool HasHardwareCastV =
    IsComplete<HardwareCastImpl<kDev, Dst, Src>>::value;

// Intermediate type used for fallback conversions, usually `float`.
template <Device::Type kDev>
struct CastBridge {
  using Type = float;
};

template <Device::Type kDev>
struct Caster {
  template <typename Dst, typename Src>
  static constexpr Dst Cast(Src&& x) {
    using PureSrc = std::remove_cv_t<std::remove_reference_t<Src>>;
    using PureDst = std::remove_cv_t<std::remove_reference_t<Dst>>;
    using Bridge = typename CastBridge<kDev>::Type;

    if constexpr (std::is_same_v<PureSrc, PureDst>) {
      return std::forward<Src>(x);
    } else if constexpr (HasHardwareCastV<kDev, PureDst, PureSrc>) {
      return HardwareCastImpl<kDev, PureDst, PureSrc>::Apply(
          std::forward<Src>(x));
    } else if constexpr (!std::is_same_v<PureSrc, Bridge> &&
                         !std::is_same_v<PureDst, Bridge> &&
                         HasHardwareCastV<kDev, Bridge, PureSrc> &&
                         (HasHardwareCastV<kDev, PureDst, Bridge> ||
                          std::is_arithmetic_v<PureDst>)) {
      Bridge tmp = Cast<Bridge>(std::forward<Src>(x));
      return Cast<PureDst>(tmp);
    } else if constexpr (!std::is_same_v<PureDst, Bridge> &&
                         std::is_arithmetic_v<PureSrc> &&
                         HasHardwareCastV<kDev, PureDst, Bridge>) {
      Bridge tmp = static_cast<Bridge>(std::forward<Src>(x));
      return Cast<PureDst>(tmp);
    } else if constexpr (std::is_arithmetic_v<PureSrc> &&
                         std::is_arithmetic_v<PureDst>) {
      return static_cast<PureDst>(std::forward<Src>(x));
    } else {
      static_assert(HasHardwareCastV<kDev, PureDst, PureSrc>,
                    "no cast path available. "
                    "Need to provide `HardwareCastImpl` specialization.");
    }
  }
};

// Convenience wrapper for casting between two types.
// Otherwise, need to write `Caster<kDev>::template Cast<Target>(val)` every
// time.
template <Device::Type kDev, typename Target, typename Source>
inline Target CastTo(Source&& val) {
  return Caster<kDev>::template Cast<Target>(std::forward<Source>(val));
}

template <Device::Type kDev, typename Source>
inline float ToFloat(Source&& val) {
  return Caster<kDev>::template Cast<float>(std::forward<Source>(val));
}

}  // namespace infini::ccl

#endif  // INFINI_CCL_CASTER_H_
