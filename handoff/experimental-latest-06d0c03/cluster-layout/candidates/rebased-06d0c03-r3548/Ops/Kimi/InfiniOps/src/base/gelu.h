#ifndef INFINI_OPS_BASE_GELU_H_
#define INFINI_OPS_BASE_GELU_H_

#include <cassert>
#include <string>

#include "operator.h"

namespace infini::ops {

class Gelu : public Operator<Gelu> {
 public:
  Gelu(const Tensor input, const std::string approximate, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        approximate_{approximate},
        output_size_{out.numel()},
        ndim_{out.ndim()},
        is_input_contiguous_{input.IsContiguous()},
        is_out_contiguous_{out.IsContiguous()},
        device_index_{out.device().index()} {
    assert((approximate == "none" || approximate == "tanh") &&
           "`Gelu` requires `approximate` to be `none` or `tanh`");
  }

  virtual void operator()(const Tensor input, const std::string approximate,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  std::string approximate_{};

  Tensor::Size output_size_{0};

  Tensor::Size ndim_{0};

  bool is_input_contiguous_{false};

  bool is_out_contiguous_{false};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
