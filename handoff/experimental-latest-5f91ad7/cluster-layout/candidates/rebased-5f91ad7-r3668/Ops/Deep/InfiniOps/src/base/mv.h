#ifndef INFINI_OPS_BASE_MV_H_
#define INFINI_OPS_BASE_MV_H_

#include "operator.h"

namespace infini::ops {

class Mv : public Operator<Mv> {
 public:
  Mv(const Tensor input, const Tensor vec, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        vec_shape_{vec.shape()},
        vec_strides_{vec.strides()},
        vec_type_{vec.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const Tensor vec,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape vec_shape_;

  Tensor::Strides vec_strides_;

  DataType vec_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif
