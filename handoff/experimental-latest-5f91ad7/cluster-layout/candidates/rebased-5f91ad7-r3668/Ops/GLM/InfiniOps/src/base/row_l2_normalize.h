#ifndef INFINI_OPS_BASE_ROW_L2_NORMALIZE_H_
#define INFINI_OPS_BASE_ROW_L2_NORMALIZE_H_

#include <cstdint>
#include <cstring>
#include <functional>

#include "operator.h"

namespace infini::ops {

class RowL2Normalize : public Operator<RowL2Normalize> {
 public:
  static constexpr bool kAutomaticDefault = true;

  RowL2Normalize(const Tensor input, const double eps, Tensor out)
      : input_meta_(input),
        out_meta_(out),
        device_(input.device()),
        eps_(eps) {}

  virtual void operator()(const Tensor input, const double eps,
                          Tensor out) const = 0;

 protected:
  const Tensor input_meta_, out_meta_;
  Device device_;
  double eps_;
};

template <>
struct CacheKeyBuilder<RowL2Normalize> {
  detail::CacheKey operator()(const Config& config, const Tensor input,
                              const double eps, Tensor out) const {
    std::uint64_t bits;
    static_assert(sizeof(bits) == sizeof(eps));
    std::memcpy(&bits, &eps, sizeof(bits));
    return detail::CacheKey::Build(config.implementation_index(), input, bits,
                                   out);
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_ROW_L2_NORMALIZE_H_
