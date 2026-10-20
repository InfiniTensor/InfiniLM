#ifndef INFINI_OPS_BASE_PAD_CONSTANT_H_
#define INFINI_OPS_BASE_PAD_CONSTANT_H_

#include <cstdint>
#include <vector>

#include "operator.h"

namespace infini::ops {

// Constant padding in Torch's trailing-dimension pair order. This is an
// inference primitive, not the old generic reflect/replicate/circular Pad.
class PadConstant : public Operator<PadConstant> {
 public:
  static constexpr bool kAutomaticDefault = true;

  PadConstant(const Tensor input, const std::vector<std::int64_t> padding,
              const double value, Tensor out)
      : input_shape_(input.shape()), input_strides_(input.strides()),
        out_shape_(out.shape()), out_strides_(out.strides()),
        dtype_(input.dtype()), device_(input.device()),
        padding_(padding), value_(value) {}

  virtual void operator()(const Tensor input,
                          const std::vector<std::int64_t> padding,
                          const double value, Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_, out_shape_;
  Tensor::Strides input_strides_, out_strides_;
  DataType dtype_;
  Device device_;
  std::vector<std::int64_t> padding_;
  double value_;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_PAD_CONSTANT_H_
