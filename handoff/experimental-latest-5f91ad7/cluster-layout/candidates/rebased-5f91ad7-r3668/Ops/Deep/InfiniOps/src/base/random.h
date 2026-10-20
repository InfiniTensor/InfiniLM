#ifndef INFINI_OPS_BASE_RANDOM_H_
#define INFINI_OPS_BASE_RANDOM_H_

#include <optional>

#include "operator.h"

namespace infini::ops {

class Random : public Operator<Random> {
 public:
  Random(const Tensor input, const int64_t from,
         const std::optional<int64_t> to, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        from_{from},
        to_{to},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {}

  Random(const Tensor input, const int64_t to, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const int64_t from,
                          const std::optional<int64_t> to,
                          Tensor out) const = 0;

  virtual void operator()(const Tensor input, const int64_t to,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  int64_t from_{};

  std::optional<int64_t> to_{};

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif
