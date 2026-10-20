#ifndef INFINI_OPS_BASE_INTERNAL_LOG_SOFTMAX_H_
#define INFINI_OPS_BASE_INTERNAL_LOG_SOFTMAX_H_

#include "operator.h"

namespace infini::ops::internal {

class LogSoftmax : public Operator<LogSoftmax> {
 public:
  LogSoftmax(const Tensor input, const int64_t dim, const bool half_to_float,
             Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        dim_{dim},
        half_to_float_{half_to_float},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const int64_t dim,
                          const bool half_to_float, Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  int64_t dim_{};

  bool half_to_float_{};

  int device_index_{0};
};

}  // namespace infini::ops::internal

#endif
