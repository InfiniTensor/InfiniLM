#ifndef INFINI_OPS_BASE_CLIP_H_
#define INFINI_OPS_BASE_CLIP_H_

#include <optional>

#include "operator.h"

namespace infini::ops {

class Clip : public Operator<Clip> {
 public:
  Clip(const Tensor input, const std::optional<double> min,
       const std::optional<double> max, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        min_{min},
        max_{max},
        device_index_{out.device().index()} {}

  Clip(const Tensor input, const std::optional<Tensor> min,
       const std::optional<Tensor> max, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        has_min_{min.has_value()},
        min_shape_{min ? Tensor::Shape{min->shape()} : Tensor::Shape{}},
        min_strides_{min ? Tensor::Strides{min->strides()} : Tensor::Strides{}},
        min_type_{min ? min->dtype() : DataType::kFloat32},
        has_max_{max.has_value()},
        max_shape_{max ? Tensor::Shape{max->shape()} : Tensor::Shape{}},
        max_strides_{max ? Tensor::Strides{max->strides()} : Tensor::Strides{}},
        max_type_{max ? max->dtype() : DataType::kFloat32},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const std::optional<double> min,
                          const std::optional<double> max,
                          Tensor out) const = 0;

  virtual void operator()(const Tensor input, const std::optional<Tensor> min,
                          const std::optional<Tensor> max,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  std::optional<double> min_{};

  std::optional<double> max_{};

  bool has_min_{false};

  Tensor::Shape min_shape_;

  Tensor::Strides min_strides_;

  DataType min_type_{DataType::kFloat32};

  bool has_max_{false};

  Tensor::Shape max_shape_;

  Tensor::Strides max_strides_;

  DataType max_type_{DataType::kFloat32};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
