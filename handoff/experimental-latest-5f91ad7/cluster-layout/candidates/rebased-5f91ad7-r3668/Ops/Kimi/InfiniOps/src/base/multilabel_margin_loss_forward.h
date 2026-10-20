#ifndef INFINI_OPS_BASE_MULTILABEL_MARGIN_LOSS_FORWARD_H_
#define INFINI_OPS_BASE_MULTILABEL_MARGIN_LOSS_FORWARD_H_

#include "operator.h"

namespace infini::ops {

class MultilabelMarginLossForward
    : public Operator<MultilabelMarginLossForward> {
 public:
  MultilabelMarginLossForward(const Tensor input, const Tensor target,
                              const int64_t reduction, Tensor output,
                              Tensor is_target)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        target_shape_{target.shape()},
        target_strides_{target.strides()},
        target_type_{target.dtype()},
        output_shape_{output.shape()},
        output_strides_{output.strides()},
        output_type_{output.dtype()},
        is_target_shape_{is_target.shape()},
        is_target_strides_{is_target.strides()},
        is_target_type_{is_target.dtype()},
        reduction_{reduction},
        device_index_{output.device().index()} {}

  virtual void operator()(const Tensor input, const Tensor target,
                          const int64_t reduction, Tensor output,
                          Tensor is_target) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape target_shape_;

  Tensor::Strides target_strides_;

  DataType target_type_;

  Tensor::Shape output_shape_;

  Tensor::Strides output_strides_;

  DataType output_type_;

  Tensor::Shape is_target_shape_;

  Tensor::Strides is_target_strides_;

  DataType is_target_type_;

  int64_t reduction_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
