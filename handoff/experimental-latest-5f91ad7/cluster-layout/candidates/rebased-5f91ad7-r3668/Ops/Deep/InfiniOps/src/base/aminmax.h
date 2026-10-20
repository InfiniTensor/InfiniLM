#ifndef INFINI_OPS_BASE_AMINMAX_H_
#define INFINI_OPS_BASE_AMINMAX_H_

#include <optional>

#include "operator.h"

namespace infini::ops {

class Aminmax : public Operator<Aminmax> {
 public:
  Aminmax(const Tensor input, const std::optional<int64_t> dim,
          const bool keepdim, Tensor min, Tensor max)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        min_shape_{min.shape()},
        min_strides_{min.strides()},
        min_type_{min.dtype()},
        max_shape_{max.shape()},
        max_strides_{max.strides()},
        max_type_{max.dtype()},
        dim_{dim},
        keepdim_{keepdim},
        device_index_{min.device().index()} {}

  virtual void operator()(const Tensor input, const std::optional<int64_t> dim,
                          const bool keepdim, Tensor min, Tensor max) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape min_shape_;

  Tensor::Strides min_strides_;

  DataType min_type_;

  Tensor::Shape max_shape_;

  Tensor::Strides max_strides_;

  DataType max_type_;

  std::optional<int64_t> dim_{};

  bool keepdim_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
