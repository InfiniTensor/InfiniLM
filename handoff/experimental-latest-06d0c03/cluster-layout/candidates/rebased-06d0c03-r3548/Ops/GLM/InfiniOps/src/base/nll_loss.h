#ifndef INFINI_OPS_BASE_NLL_LOSS_H_
#define INFINI_OPS_BASE_NLL_LOSS_H_

#include <optional>
#include <string>

#include "common/op_utils/reduction.h"
#include "operator.h"

namespace infini::ops {

class NllLoss : public Operator<NllLoss> {
 public:
  NllLoss(const Tensor input, const Tensor target,
          const std::optional<Tensor> weight,
          const std::optional<bool> size_average, const int64_t ignore_index,
          const std::optional<bool> reduce, const std::string reduction,
          Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        target_shape_{target.shape()},
        target_strides_{target.strides()},
        target_type_{target.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        has_weight_{weight.has_value()},
        weight_shape_{weight ? Tensor::Shape{weight->shape()}
                             : Tensor::Shape{}},
        weight_strides_{weight ? Tensor::Strides{weight->strides()}
                               : Tensor::Strides{}},
        weight_type_{weight ? weight->dtype() : DataType::kFloat32},
        reduction_{reduction_detail::FromPythonArguments(size_average, reduce,
                                                         reduction)},
        ignore_index_{ignore_index},
        device_index_{out.device().index()} {}

  [[deprecated(
      "Use the overload with `ignore_index` before string `reduction`.")]]
  NllLoss(const Tensor input, const Tensor target,
          const std::optional<Tensor> weight, const int64_t reduction,
          const int64_t ignore_index, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        target_shape_{target.shape()},
        target_strides_{target.strides()},
        target_type_{target.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        has_weight_{weight.has_value()},
        weight_shape_{weight ? Tensor::Shape{weight->shape()}
                             : Tensor::Shape{}},
        weight_strides_{weight ? Tensor::Strides{weight->strides()}
                               : Tensor::Strides{}},
        weight_type_{weight ? weight->dtype() : DataType::kFloat32},
        reduction_{reduction},
        ignore_index_{ignore_index},
        device_index_{out.device().index()} {}

  void operator()(const Tensor input, const Tensor target,
                  const std::optional<Tensor> weight,
                  const std::optional<bool> size_average,
                  const int64_t ignore_index, const std::optional<bool> reduce,
                  const std::string reduction, Tensor out) const {
    return operator()(
        input, target, weight,
        reduction_detail::FromPythonArguments(size_average, reduce, reduction),
        ignore_index, out);
  }

  [[deprecated(
      "Use the overload with `ignore_index` before string `reduction`.")]]
  virtual void operator()(const Tensor input, const Tensor target,
                          const std::optional<Tensor> weight,
                          const int64_t reduction, const int64_t ignore_index,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape target_shape_;

  Tensor::Strides target_strides_;

  DataType target_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

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
