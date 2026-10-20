#ifndef INFINI_OPS_BASE_NATIVE_BATCH_NORM_H_
#define INFINI_OPS_BASE_NATIVE_BATCH_NORM_H_

#include <optional>

#include "operator.h"

namespace infini::ops {

class NativeBatchNorm : public Operator<NativeBatchNorm> {
 public:
  NativeBatchNorm(const Tensor input, const std::optional<Tensor> weight,
                  const std::optional<Tensor> bias,
                  const std::optional<Tensor> running_mean,
                  const std::optional<Tensor> running_var, const bool training,
                  const double momentum, const double eps, Tensor out,
                  Tensor save_mean, Tensor save_invstd)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        save_mean_shape_{save_mean.shape()},
        save_mean_strides_{save_mean.strides()},
        save_mean_type_{save_mean.dtype()},
        save_invstd_shape_{save_invstd.shape()},
        save_invstd_strides_{save_invstd.strides()},
        save_invstd_type_{save_invstd.dtype()},
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
        has_running_mean_{running_mean.has_value()},
        running_mean_shape_{running_mean ? Tensor::Shape{running_mean->shape()}
                                         : Tensor::Shape{}},
        running_mean_strides_{running_mean
                                  ? Tensor::Strides{running_mean->strides()}
                                  : Tensor::Strides{}},
        running_mean_type_{running_mean ? running_mean->dtype()
                                        : DataType::kFloat32},
        has_running_var_{running_var.has_value()},
        running_var_shape_{running_var ? Tensor::Shape{running_var->shape()}
                                       : Tensor::Shape{}},
        running_var_strides_{running_var
                                 ? Tensor::Strides{running_var->strides()}
                                 : Tensor::Strides{}},
        running_var_type_{running_var ? running_var->dtype()
                                      : DataType::kFloat32},
        training_{training},
        momentum_{momentum},
        eps_{eps},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input,
                          const std::optional<Tensor> weight,
                          const std::optional<Tensor> bias,
                          const std::optional<Tensor> running_mean,
                          const std::optional<Tensor> running_var,
                          const bool training, const double momentum,
                          const double eps, Tensor out, Tensor save_mean,
                          Tensor save_invstd) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  Tensor::Shape save_mean_shape_;

  Tensor::Strides save_mean_strides_;

  DataType save_mean_type_;

  Tensor::Shape save_invstd_shape_;

  Tensor::Strides save_invstd_strides_;

  DataType save_invstd_type_;

  bool has_weight_{false};

  Tensor::Shape weight_shape_;

  Tensor::Strides weight_strides_;

  DataType weight_type_{DataType::kFloat32};

  bool has_bias_{false};

  Tensor::Shape bias_shape_;

  Tensor::Strides bias_strides_;

  DataType bias_type_{DataType::kFloat32};

  bool has_running_mean_{false};

  Tensor::Shape running_mean_shape_;

  Tensor::Strides running_mean_strides_;

  DataType running_mean_type_{DataType::kFloat32};

  bool has_running_var_{false};

  Tensor::Shape running_var_shape_;

  Tensor::Strides running_var_strides_;

  DataType running_var_type_{DataType::kFloat32};

  bool training_{};

  double momentum_{};

  double eps_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
