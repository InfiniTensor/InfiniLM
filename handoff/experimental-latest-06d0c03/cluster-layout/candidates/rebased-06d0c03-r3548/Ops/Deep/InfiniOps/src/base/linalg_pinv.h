#ifndef INFINI_OPS_BASE_LINALG_PINV_H_
#define INFINI_OPS_BASE_LINALG_PINV_H_

#include <optional>

#include "operator.h"

namespace infini::ops::linalg {

class Pinv : public Operator<Pinv> {
 public:
  Pinv(const Tensor input, const double rcond, const bool hermitian, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        rcond_{rcond},
        hermitian_{hermitian},
        device_index_{out.device().index()} {}

  Pinv(const Tensor input, const std::optional<Tensor> atol,
       const std::optional<Tensor> rtol, const bool hermitian, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        has_atol_{atol.has_value()},
        atol_shape_{atol ? Tensor::Shape{atol->shape()} : Tensor::Shape{}},
        atol_strides_{atol ? Tensor::Strides{atol->strides()}
                           : Tensor::Strides{}},
        atol_type_{atol ? atol->dtype() : DataType::kFloat32},
        has_rtol_{rtol.has_value()},
        rtol_shape_{rtol ? Tensor::Shape{rtol->shape()} : Tensor::Shape{}},
        rtol_strides_{rtol ? Tensor::Strides{rtol->strides()}
                           : Tensor::Strides{}},
        rtol_type_{rtol ? rtol->dtype() : DataType::kFloat32},
        hermitian_{hermitian},
        device_index_{out.device().index()} {}

  Pinv(const Tensor input, const std::optional<double> atol,
       const std::optional<double> rtol, const bool hermitian, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        hermitian_{hermitian},
        atol_{atol},
        rtol_{rtol},
        device_index_{out.device().index()} {}

  Pinv(const Tensor input, const Tensor rcond, const bool hermitian, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        rcond_shape_{rcond.shape()},
        rcond_strides_{rcond.strides()},
        rcond_type_{rcond.dtype()},
        hermitian_{hermitian},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const double rcond,
                          const bool hermitian, Tensor out) const = 0;

  virtual void operator()(const Tensor input, const std::optional<Tensor> atol,
                          const std::optional<Tensor> rtol,
                          const bool hermitian, Tensor out) const = 0;

  virtual void operator()(const Tensor input, const std::optional<double> atol,
                          const std::optional<double> rtol,
                          const bool hermitian, Tensor out) const = 0;

  virtual void operator()(const Tensor input, const Tensor rcond,
                          const bool hermitian, Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  double rcond_{};

  bool hermitian_{};

  bool has_atol_{false};

  Tensor::Shape atol_shape_;

  Tensor::Strides atol_strides_;

  DataType atol_type_{DataType::kFloat32};

  bool has_rtol_{false};

  Tensor::Shape rtol_shape_;

  Tensor::Strides rtol_strides_;

  DataType rtol_type_{DataType::kFloat32};

  std::optional<double> atol_{};

  std::optional<double> rtol_{};

  Tensor::Shape rcond_shape_;

  Tensor::Strides rcond_strides_;

  DataType rcond_type_;

  int device_index_{0};
};

}  // namespace infini::ops::linalg

#endif
