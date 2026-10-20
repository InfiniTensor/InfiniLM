#ifndef INFINI_OPS_BASE_TAKE_ALONG_DIM_H_
#define INFINI_OPS_BASE_TAKE_ALONG_DIM_H_

#include <optional>

#include "operator.h"

namespace infini::ops {

class TakeAlongDim : public Operator<TakeAlongDim> {
 public:
  TakeAlongDim(const Tensor input, const Tensor indices,
               const std::optional<int64_t> dim, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        indices_shape_{indices.shape()},
        indices_strides_{indices.strides()},
        indices_type_{indices.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        dim_{dim},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const Tensor indices,
                          const std::optional<int64_t> dim,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape indices_shape_;

  Tensor::Strides indices_strides_;

  DataType indices_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  std::optional<int64_t> dim_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
