#ifndef INFINI_OPS_BASE_DIFF_H_
#define INFINI_OPS_BASE_DIFF_H_

#include <optional>

#include "operator.h"

namespace infini::ops {

class Diff : public Operator<Diff> {
 public:
  Diff(const Tensor input, const std::optional<Tensor> prepend,
       const std::optional<Tensor> append, const int64_t n, const int64_t dim,
       Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        has_prepend_{prepend.has_value()},
        prepend_shape_{prepend ? Tensor::Shape{prepend->shape()}
                               : Tensor::Shape{}},
        prepend_strides_{prepend ? Tensor::Strides{prepend->strides()}
                                 : Tensor::Strides{}},
        prepend_type_{prepend ? prepend->dtype() : DataType::kFloat32},
        has_append_{append.has_value()},
        append_shape_{append ? Tensor::Shape{append->shape()}
                             : Tensor::Shape{}},
        append_strides_{append ? Tensor::Strides{append->strides()}
                               : Tensor::Strides{}},
        append_type_{append ? append->dtype() : DataType::kFloat32},
        n_{n},
        dim_{dim},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input,
                          const std::optional<Tensor> prepend,
                          const std::optional<Tensor> append, const int64_t n,
                          const int64_t dim, Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  bool has_prepend_{false};

  Tensor::Shape prepend_shape_;

  Tensor::Strides prepend_strides_;

  DataType prepend_type_{DataType::kFloat32};

  bool has_append_{false};

  Tensor::Shape append_shape_;

  Tensor::Strides append_strides_;

  DataType append_type_{DataType::kFloat32};

  int64_t n_{};

  int64_t dim_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
