#ifndef INFINI_OPS_BASE_MSE_LOSS_H_
#define INFINI_OPS_BASE_MSE_LOSS_H_

#include <cassert>
#include <optional>
#include <string>

#include "common/op_utils/reduction.h"
#include "operator.h"

namespace infini::ops {

class MseLoss : public Operator<MseLoss> {
 public:
  MseLoss(const Tensor input, const Tensor target,
          const std::optional<Tensor> weight,
          const std::optional<bool> size_average,
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
        reduction_{reduction_detail::FromPythonArguments(size_average, reduce,
                                                         reduction)},
        device_index_{out.device().index()} {
    assert(!weight.has_value() &&
           "`weight` is unsupported because the current ATen `MseLoss` ABI "
           "cannot implement the Python wrapper's weighted composition");
  }

  /// \deprecated Use the PyTorch-compatible overload. This constructor will
  /// be removed in a future release.
  [[deprecated("Use the PyTorch-compatible overload instead.")]]
  MseLoss(const Tensor input, const Tensor target, const int64_t reduction,
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
        reduction_{reduction},
        device_index_{out.device().index()} {}

  void operator()(const Tensor input, const Tensor target,
                  const std::optional<Tensor> weight,
                  const std::optional<bool> size_average,
                  const std::optional<bool> reduce, const std::string reduction,
                  Tensor out) const {
    assert(!weight.has_value() &&
           "`weight` is unsupported because the current ATen `MseLoss` ABI "
           "cannot implement the Python wrapper's weighted composition");

    return operator()(
        input, target,
        reduction_detail::FromPythonArguments(size_average, reduce, reduction),
        out);
  }

  /// \deprecated Use the PyTorch-compatible overload. This overload will be
  /// removed in a future release.
  [[deprecated("Use the PyTorch-compatible overload instead.")]]
  virtual void operator()(const Tensor input, const Tensor target,
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

  int64_t reduction_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
