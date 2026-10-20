#ifndef INFINI_OPS_BASE_CAST_H_
#define INFINI_OPS_BASE_CAST_H_

#include "operator.h"

namespace infini::ops {

class Cast : public Operator<Cast> {
 public:
  Cast(const Tensor input, Tensor out)
      : ndim_{out.ndim()},
        output_size_{out.numel()},
        input_dtype_{input.dtype()},
        out_dtype_{out.dtype()},
        input_shape_{input.shape()},
        out_shape_{out.shape()},
        input_strides_{input.strides()},
        out_strides_{out.strides()},
        is_input_contiguous_{input.IsContiguous()},
        is_out_contiguous_{out.IsContiguous()} {
    assert(input.numel() == out.numel() &&
           "the input and output of `Cast` must have the same number of "
           "elements");
  }

  virtual void operator()(const Tensor input, Tensor out) const = 0;

 protected:
  Tensor::Size ndim_{0};

  Tensor::Size output_size_{0};

  const DataType input_dtype_;

  const DataType out_dtype_;

  Tensor::Shape input_shape_;

  Tensor::Shape out_shape_;

  Tensor::Strides input_strides_;

  Tensor::Strides out_strides_;

  bool is_input_contiguous_{false};

  bool is_out_contiguous_{false};
};

}  // namespace infini::ops

#endif
