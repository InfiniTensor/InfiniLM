#ifndef INFINI_OPS_BASE_BATCH_NORM_ELEMT_H_
#define INFINI_OPS_BASE_BATCH_NORM_ELEMT_H_

#include <optional>

#include "operator.h"

namespace infini::ops {

class BatchNormElemt : public Operator<BatchNormElemt> {
 public:
  BatchNormElemt(const Tensor input, const std::optional<Tensor> weight,
                 const std::optional<Tensor> bias, const Tensor mean,
                 const Tensor invstd, const double eps, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        mean_shape_{mean.shape()},
        mean_strides_{mean.strides()},
        mean_type_{mean.dtype()},
        invstd_shape_{invstd.shape()},
        invstd_strides_{invstd.strides()},
        invstd_type_{invstd.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        has_weight_{weight.has_value()},
        weight_shape_{weight ? Tensor::Shape{weight->shape()}
                             : Tensor::Shape{}},
        weight_strides_{weight ? Tensor::Strides{weight->strides()}
                               : Tensor::Strides{}},
        weight_type_{weight ? weight->dtype() : DataType::kFloat32},
        has_bias_{bias.has_value()},
        bias_shape_{bias ? Tensor::Shape{bias->shape()} : Tensor::Shape{}},
        bias_strides_{bias ? Tensor::Strides{bias->strides()}
                           : Tensor::Strides{}},
        bias_type_{bias ? bias->dtype() : DataType::kFloat32},
        eps_{eps},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input,
                          const std::optional<Tensor> weight,
                          const std::optional<Tensor> bias, const Tensor mean,
                          const Tensor invstd, const double eps,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape mean_shape_;

  Tensor::Strides mean_strides_;

  DataType mean_type_;

  Tensor::Shape invstd_shape_;

  Tensor::Strides invstd_strides_;

  DataType invstd_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  bool has_weight_{false};

  Tensor::Shape weight_shape_;

  Tensor::Strides weight_strides_;

  DataType weight_type_{DataType::kFloat32};

  bool has_bias_{false};

  Tensor::Shape bias_shape_;

  Tensor::Strides bias_strides_;

  DataType bias_type_{DataType::kFloat32};

  double eps_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
