#ifndef INFINI_OPS_CUDA_CASTER_CUH_
#define INFINI_OPS_CUDA_CASTER_CUH_

#include "caster.h"
#include "data_type.h"

namespace infini::ops {

namespace detail {

template <Device::Type kDev, typename T>
struct ToFloat;

template <Device::Type kDev, typename T>
struct FromFloat;

template <Device::Type kDev, typename Dst, typename Src>
struct HardwareCast {
  inline static constexpr bool kSupported = false;
};

}  // namespace detail

template <Device::Type kDev>
struct CudaCasterImpl {
  template <typename Dst, typename Src>
  __host__ __device__ static Dst Cast(Src&& x) {
    static_assert(!std::is_reference_v<Dst>,
                  "`Cast` cannot return reference types");

    using PureSrc = std::remove_cv_t<std::remove_reference_t<Src>>;
    using PureDst = std::remove_cv_t<std::remove_reference_t<Dst>>;

    if constexpr (std::is_same_v<PureSrc, PureDst>) {
      return std::forward<Src>(x);
    } else {
      return HardwareCast<PureDst>(std::forward<Src>(x), PriorityHigh{});
    }
  }

 private:
  template <typename T>
  using PureType = std::remove_cv_t<std::remove_reference_t<T>>;

  template <typename T>
  __host__ __device__ static constexpr float ToFloatHelper(T&& x) {
    using PureSrc = PureType<T>;
    if constexpr (IsBFloat16<kDev, PureSrc> || IsFP16<kDev, PureSrc>) {
      return detail::ToFloat<kDev, PureSrc>{}(std::forward<T>(x));
    } else {
      return static_cast<float>(std::forward<T>(x));
    }
  }

  template <typename Dst>
  __host__ __device__ static constexpr Dst FromFloatHelper(float f) {
    using PureDst = PureType<Dst>;
    if constexpr (IsBFloat16<kDev, PureDst> || IsFP16<kDev, PureDst>) {
      return detail::FromFloat<kDev, PureDst>{}(f);
    } else {
      return static_cast<Dst>(f);
    }
  }

  struct PriorityLow {};
  struct PriorityHigh : PriorityLow {};

  template <typename Dst, typename Src>
  __host__ __device__ static constexpr Dst HardwareCast(Src&& x, PriorityLow) {
    return FromFloatHelper<Dst>(ToFloatHelper(std::forward<Src>(x)));
  }

  template <typename Dst, typename Src>
  __host__ __device__ static auto HardwareCast(Src x, PriorityHigh)
      -> std::enable_if_t<
          detail::HardwareCast<kDev, PureType<Dst>, PureType<Src>>::kSupported,
          Dst> {
    return detail::HardwareCast<kDev, PureType<Dst>, PureType<Src>>{}(x);
  }
};

}  // namespace infini::ops

#endif
