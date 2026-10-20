#ifndef INFINI_OPS_BASE_CUMSUM_H_
#define INFINI_OPS_BASE_CUMSUM_H_

#include <optional>

#include "operator.h"

namespace infini::ops {

class Cumsum : public Operator<Cumsum> {
 public:
  Cumsum(const Tensor input, const int64_t dim,
         const std::optional<DataType> dtype, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        dim_{dim},
        dtype_{dtype},
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

  int device_index_{0};
};

}  // namespace infini::ops

#endif
