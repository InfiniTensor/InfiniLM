#ifndef INFINI_OPS_BASE_ROW_L2_NORMALIZE_H_
#define INFINI_OPS_BASE_ROW_L2_NORMALIZE_H_

#include <cstdint>
#include <cstring>

#include "operator.h"

namespace infini::ops {

class RowL2Normalize : public Operator<RowL2Normalize> {
 public:
  static constexpr bool kAutomaticDefault = true;

  RowL2Normalize(const Tensor input, const double eps, Tensor out)
      : input_shape_(input.shape()), input_strides_(input.strides()),
        out_shape_(out.shape()), out_strides_(out.strides()),
        dtype_(input.dtype()), device_(input.device()), eps_(eps) {}

  virtual void operator()(const Tensor input, const double eps,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_, out_shape_;
  Tensor::Strides input_strides_, out_strides_;
  DataType dtype_;
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
    return detail::CacheKey::Build(config.implementation_index(), input,
                                   bits, out);
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_ROW_L2_NORMALIZE_H_
