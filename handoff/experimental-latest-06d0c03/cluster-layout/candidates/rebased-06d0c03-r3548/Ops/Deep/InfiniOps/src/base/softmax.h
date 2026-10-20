#ifndef INFINI_OPS_BASE_SOFTMAX_H_
#define INFINI_OPS_BASE_SOFTMAX_H_

#include <optional>

#include "operator.h"

namespace infini::ops {

class Softmax : public Operator<Softmax> {
 public:
  Softmax(const Tensor input, const int64_t dim,
          const std::optional<DataType> dtype, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        dim_{dim < 0 ? dim + static_cast<int64_t>(input.ndim()) : dim},
        dtype_{dtype},
        ndim_{out.ndim()},
        dim_size_{out.size(dim_)},
        row_count_{dim_size_ == 0 ? 0 : out.numel() / dim_size_},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const int64_t dim,
                          const std::optional<DataType> dtype,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  int64_t dim_{};

  std::optional<DataType> dtype_{};

  Tensor::Size ndim_{0};

  Tensor::Size dim_size_{0};

  Tensor::Size row_count_{0};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
