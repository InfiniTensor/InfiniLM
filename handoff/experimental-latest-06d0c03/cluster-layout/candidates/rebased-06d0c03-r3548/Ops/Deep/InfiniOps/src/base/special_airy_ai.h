#ifndef INFINI_OPS_BASE_SPECIAL_AIRY_AI_H_
#define INFINI_OPS_BASE_SPECIAL_AIRY_AI_H_

#include "operator.h"

namespace infini::ops::special {

class AiryAi : public Operator<AiryAi> {
 public:
  AiryAi(const Tensor x, Tensor out)
      : x_shape_{x.shape()},
        x_strides_{x.strides()},
        x_type_{x.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor x, Tensor out) const = 0;

 protected:
  Tensor::Shape x_shape_;

  Tensor::Strides x_strides_;

  DataType x_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  int device_index_{0};
};

}  // namespace infini::ops::special

#endif
