#ifndef INFINI_OPS_BASE_HUBER_LOSS_BACKWARD_H_
#define INFINI_OPS_BASE_HUBER_LOSS_BACKWARD_H_

#include "operator.h"

namespace infini::ops {

class HuberLossBackward : public Operator<HuberLossBackward> {
 public:
  HuberLossBackward(const Tensor grad_output, const Tensor input,
                    const Tensor target, const int64_t reduction,
                    const double delta, Tensor grad_input)
      : grad_output_shape_{grad_output.shape()},
        grad_output_strides_{grad_output.strides()},
        grad_output_type_{grad_output.dtype()},
        input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        target_shape_{target.shape()},
        target_strides_{target.strides()},
        target_type_{target.dtype()},
        grad_input_shape_{grad_input.shape()},
        grad_input_strides_{grad_input.strides()},
        grad_input_type_{grad_input.dtype()},
        reduction_{reduction},
        delta_{delta},
        device_index_{grad_input.device().index()} {}

  virtual void operator()(const Tensor grad_output, const Tensor input,
                          const Tensor target, const int64_t reduction,
                          const double delta, Tensor grad_input) const = 0;

 protected:
  Tensor::Shape grad_output_shape_;

  Tensor::Strides grad_output_strides_;

  DataType grad_output_type_;

  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape target_shape_;

  Tensor::Strides target_strides_;

  DataType target_type_;

  Tensor::Shape grad_input_shape_;

  Tensor::Strides grad_input_strides_;

  DataType grad_input_type_;

  int64_t reduction_{};

  double delta_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
