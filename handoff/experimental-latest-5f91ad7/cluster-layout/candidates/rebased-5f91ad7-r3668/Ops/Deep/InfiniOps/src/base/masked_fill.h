#ifndef INFINI_OPS_BASE_MASKED_FILL_H_
#define INFINI_OPS_BASE_MASKED_FILL_H_

#include "operator.h"

namespace infini::ops {

class MaskedFill : public Operator<MaskedFill> {
 public:
  MaskedFill(const Tensor input, const Tensor mask, const double value,
             Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        mask_shape_{mask.shape()},
        mask_strides_{mask.strides()},
        mask_type_{mask.dtype()},
        value_{value},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {}

  MaskedFill(const Tensor input, const Tensor mask, const Tensor value,
             Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        mask_shape_{mask.shape()},
        mask_strides_{mask.strides()},
        mask_type_{mask.dtype()},
        value_shape_{value.shape()},
        value_strides_{value.strides()},
        value_type_{value.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const Tensor mask,
                          const double value, Tensor out) const = 0;

  virtual void operator()(const Tensor input, const Tensor mask,
                          const Tensor value, Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape mask_shape_;

  Tensor::Strides mask_strides_;

  DataType mask_type_;

  double value_{};

  Tensor::Shape value_shape_;

  Tensor::Strides value_strides_;

  DataType value_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif
