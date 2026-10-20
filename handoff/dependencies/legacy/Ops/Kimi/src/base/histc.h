#ifndef INFINI_OPS_BASE_HISTC_H_
#define INFINI_OPS_BASE_HISTC_H_

#include "operator.h"

namespace infini::ops {

class Histc : public Operator<Histc> {
 public:
  Histc(const Tensor input, const int64_t bins, const double min,
        const double max, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        bins_{bins},
        min_{min},
        max_{max},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const int64_t bins,
                          const double min, const double max,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  int64_t bins_{};

  double min_{};

  double max_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
