#ifndef INFINI_OPS_BASE_ADDCDIV_H_
#define INFINI_OPS_BASE_ADDCDIV_H_

#include "operator.h"

namespace infini::ops {

class Addcdiv : public Operator<Addcdiv> {
 public:
  Addcdiv(const Tensor input, const Tensor tensor1, const Tensor tensor2,
          const double value, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        tensor1_shape_{tensor1.shape()},
        tensor1_strides_{tensor1.strides()},
        tensor1_type_{tensor1.dtype()},
        tensor2_shape_{tensor2.shape()},
        tensor2_strides_{tensor2.strides()},
        tensor2_type_{tensor2.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        value_{value},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const Tensor tensor1,
                          const Tensor tensor2, const double value,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape tensor1_shape_;

  Tensor::Strides tensor1_strides_;

  DataType tensor1_type_;

  Tensor::Shape tensor2_shape_;

  Tensor::Strides tensor2_strides_;

  DataType tensor2_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  double value_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
