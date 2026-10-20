#ifndef INFINI_OPS_BASE_HARDTANH_H_
#define INFINI_OPS_BASE_HARDTANH_H_

#include "operator.h"

namespace infini::ops {

class Hardtanh : public Operator<Hardtanh> {
 public:
  Hardtanh(const Tensor input, const double min_val, const double max_val,
           Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        min_val_{min_val},
        max_val_{max_val},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const double min_val,
                          const double max_val, Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  double min_val_{};

  double max_val_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
