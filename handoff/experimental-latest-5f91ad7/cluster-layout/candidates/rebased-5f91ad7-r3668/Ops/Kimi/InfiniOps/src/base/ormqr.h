#ifndef INFINI_OPS_BASE_ORMQR_H_
#define INFINI_OPS_BASE_ORMQR_H_

#include "operator.h"

namespace infini::ops {

class Ormqr : public Operator<Ormqr> {
 public:
  Ormqr(const Tensor input, const Tensor input2, const Tensor input3,
        const bool left, const bool transpose, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        input2_shape_{input2.shape()},
        input2_strides_{input2.strides()},
        input2_type_{input2.dtype()},
        input3_shape_{input3.shape()},
        input3_strides_{input3.strides()},
        input3_type_{input3.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        left_{left},
        transpose_{transpose},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const Tensor input2,
                          const Tensor input3, const bool left,
                          const bool transpose, Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape input2_shape_;

  Tensor::Strides input2_strides_;

  DataType input2_type_;

  Tensor::Shape input3_shape_;

  Tensor::Strides input3_strides_;

  DataType input3_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  bool left_{};

  bool transpose_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
