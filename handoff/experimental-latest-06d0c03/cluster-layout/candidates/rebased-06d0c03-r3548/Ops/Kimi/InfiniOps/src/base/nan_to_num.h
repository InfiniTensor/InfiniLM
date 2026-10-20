#ifndef INFINI_OPS_BASE_NAN_TO_NUM_H_
#define INFINI_OPS_BASE_NAN_TO_NUM_H_

#include <optional>

#include "operator.h"

namespace infini::ops {

class NanToNum : public Operator<NanToNum> {
 public:
  NanToNum(const Tensor input, const std::optional<double> nan,
           const std::optional<double> posinf,
           const std::optional<double> neginf, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        nan_{nan},
        posinf_{posinf},
        neginf_{neginf},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const std::optional<double> nan,
                          const std::optional<double> posinf,
                          const std::optional<double> neginf,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  std::optional<double> nan_{};

  std::optional<double> posinf_{};

  std::optional<double> neginf_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
