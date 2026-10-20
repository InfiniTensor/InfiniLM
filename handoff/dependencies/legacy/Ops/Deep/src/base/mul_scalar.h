#ifndef INFINI_OPS_BASE_MUL_SCALAR_H_
#define INFINI_OPS_BASE_MUL_SCALAR_H_

#include <cstdint>
#include <cstring>

#include "operator.h"

namespace infini::ops {

class MulScalar : public Operator<MulScalar> {
 public:
  static constexpr bool kAutomaticDefault = true;

  MulScalar(const Tensor input, const double alpha, Tensor out)
      : input_shape_(input.shape()), input_strides_(input.strides()),
        out_shape_(out.shape()), out_strides_(out.strides()),
        dtype_(input.dtype()), device_(input.device()), alpha_(alpha) {}

  virtual void operator()(const Tensor input, const double alpha,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_, out_shape_;
  Tensor::Strides input_strides_, out_strides_;
  DataType dtype_;
  Device device_;
  double alpha_;
};

// The scalar is an execution argument, not a device tensor. Hash its exact
// bits so a cached instance cannot substitute +0 for -0 or another NaN payload.
template <>
struct CacheKeyBuilder<MulScalar> {
  detail::CacheKey operator()(const Config& config, const Tensor input,
                              const double alpha, Tensor out) const {
    std::uint64_t bits;
    static_assert(sizeof(bits) == sizeof(alpha));
    std::memcpy(&bits, &alpha, sizeof(bits));
    return detail::CacheKey::Build(config.implementation_index(), input,
                                   bits, out);
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_MUL_SCALAR_H_
