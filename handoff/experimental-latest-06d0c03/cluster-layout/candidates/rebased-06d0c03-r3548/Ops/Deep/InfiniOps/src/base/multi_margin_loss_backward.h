#ifndef INFINI_OPS_BASE_MULTI_MARGIN_LOSS_BACKWARD_H_
#define INFINI_OPS_BASE_MULTI_MARGIN_LOSS_BACKWARD_H_

#include <optional>

#include "operator.h"

namespace infini::ops {

class MultiMarginLossBackward : public Operator<MultiMarginLossBackward> {
 public:
  MultiMarginLossBackward(const Tensor grad_output, const Tensor input,
                          const Tensor target,
                          const std::optional<Tensor> weight, const double p,
                          const double margin, const int64_t reduction,
                          Tensor grad_input)
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
        has_weight_{weight.has_value()},
        weight_shape_{weight ? Tensor::Shape{weight->shape()}
                             : Tensor::Shape{}},
        weight_strides_{weight ? Tensor::Strides{weight->strides()}
                               : Tensor::Strides{}},
        weight_type_{weight ? weight->dtype() : DataType::kFloat32},
        p_{p},
        margin_{margin},
        reduction_{reduction},
        device_index_{grad_input.device().index()} {}

  virtual void operator()(const Tensor grad_output, const Tensor input,
                          const Tensor target,
                          const std::optional<Tensor> weight, const double p,
                          const double margin, const int64_t reduction,
                          Tensor grad_input) const = 0;

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

  bool has_weight_{false};

  Tensor::Shape weight_shape_;

  Tensor::Strides weight_strides_;

  DataType weight_type_{DataType::kFloat32};

  double p_{};

  double margin_{};

  int64_t reduction_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
