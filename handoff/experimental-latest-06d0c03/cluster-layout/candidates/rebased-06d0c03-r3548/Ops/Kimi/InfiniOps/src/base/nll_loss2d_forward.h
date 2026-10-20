#ifndef INFINI_OPS_BASE_NLL_LOSS2D_FORWARD_H_
#define INFINI_OPS_BASE_NLL_LOSS2D_FORWARD_H_

#include <optional>

#include "operator.h"

namespace infini::ops {

class NllLoss2dForward : public Operator<NllLoss2dForward> {
 public:
  NllLoss2dForward(const Tensor input, const Tensor target,
                   const std::optional<Tensor> weight, const int64_t reduction,
                   const int64_t ignore_index, Tensor output,
                   Tensor total_weight)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        target_shape_{target.shape()},
        target_strides_{target.strides()},
        target_type_{target.dtype()},
        output_shape_{output.shape()},
        output_strides_{output.strides()},
        output_type_{output.dtype()},
        total_weight_shape_{total_weight.shape()},
        total_weight_strides_{total_weight.strides()},
        total_weight_type_{total_weight.dtype()},
        has_weight_{weight.has_value()},
        weight_shape_{weight ? Tensor::Shape{weight->shape()}
                             : Tensor::Shape{}},
        weight_strides_{weight ? Tensor::Strides{weight->strides()}
                               : Tensor::Strides{}},
        weight_type_{weight ? weight->dtype() : DataType::kFloat32},
        reduction_{reduction},
        ignore_index_{ignore_index},
        device_index_{output.device().index()} {}

  virtual void operator()(const Tensor input, const Tensor target,
                          const std::optional<Tensor> weight,
                          const int64_t reduction, const int64_t ignore_index,
                          Tensor output, Tensor total_weight) const = 0;

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

  Tensor::Shape total_weight_shape_;

  Tensor::Strides total_weight_strides_;

  DataType total_weight_type_;

  bool has_weight_{false};

  Tensor::Shape weight_shape_;

  Tensor::Strides weight_strides_;

  DataType weight_type_{DataType::kFloat32};

  int64_t reduction_{};

  int64_t ignore_index_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
