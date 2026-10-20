#ifndef INFINI_OPS_BASE_BINARY_CROSS_ENTROPY_H_
#define INFINI_OPS_BASE_BINARY_CROSS_ENTROPY_H_

#include <optional>
#include <string>

#include "common/op_utils/reduction.h"
#include "operator.h"

namespace infini::ops {

class BinaryCrossEntropy : public Operator<BinaryCrossEntropy> {
 public:
  BinaryCrossEntropy(const Tensor input, const Tensor target,
                     const std::optional<Tensor> weight,
                     const std::optional<bool> size_average,
                     const std::optional<bool> reduce,
                     const std::string reduction, Tensor out)
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
        device_index_{out.device().index()} {}

  /// \deprecated Use the overload with Python-compatible reduction
  /// arguments. This constructor will be removed in a future release.
  [[deprecated("Use the Python-compatible reduction overload instead.")]]
  BinaryCrossEntropy(const Tensor input, const Tensor target,
                     const std::optional<Tensor> weight,
                     const int64_t reduction, Tensor out)
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
        device_index_{out.device().index()} {}

  void operator()(const Tensor input, const Tensor target,
                  const std::optional<Tensor> weight,
                  const std::optional<bool> size_average,
                  const std::optional<bool> reduce, const std::string reduction,
                  Tensor out) const {
    (*this)(
        input, target, weight,
        reduction_detail::FromPythonArguments(size_average, reduce, reduction),
        out);
  }

  /// \deprecated Use the overload with Python-compatible reduction
  /// arguments. This overload will be removed in a future release.
  [[deprecated("Use the Python-compatible reduction overload instead.")]]
  virtual void operator()(const Tensor input, const Tensor target,
                          const std::optional<Tensor> weight,
                          const int64_t reduction, Tensor out) const = 0;

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

  int device_index_{0};
};

}  // namespace infini::ops

#endif
