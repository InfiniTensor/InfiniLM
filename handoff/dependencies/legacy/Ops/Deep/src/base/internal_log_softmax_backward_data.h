#ifndef INFINI_OPS_BASE_INTERNAL_LOG_SOFTMAX_BACKWARD_DATA_H_
#define INFINI_OPS_BASE_INTERNAL_LOG_SOFTMAX_BACKWARD_DATA_H_

#include "operator.h"

namespace infini::ops::internal {

class LogSoftmaxBackwardData : public Operator<LogSoftmaxBackwardData> {
 public:
  LogSoftmaxBackwardData(const Tensor grad_output, const Tensor output,
                         const int64_t dim, const DataType input_dtype,
                         Tensor out)
      : grad_output_shape_{grad_output.shape()},
        grad_output_strides_{grad_output.strides()},
        grad_output_type_{grad_output.dtype()},
        output_shape_{output.shape()},
        output_strides_{output.strides()},
        output_type_{output.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        dim_{dim},
        input_dtype_{input_dtype},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor grad_output, const Tensor output,
                          const int64_t dim, const DataType input_dtype,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape grad_output_shape_;

  Tensor::Strides grad_output_strides_;

  DataType grad_output_type_;

  Tensor::Shape output_shape_;

  Tensor::Strides output_strides_;

  DataType output_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  int64_t dim_{};

  DataType input_dtype_{};

  int device_index_{0};
};

}  // namespace infini::ops::internal

#endif
