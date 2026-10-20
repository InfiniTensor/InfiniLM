#ifndef INFINI_OPS_BASE_SILU_H_
#define INFINI_OPS_BASE_SILU_H_

#include <cstddef>

#include "data_type.h"
#include "operator.h"

namespace infini::ops {

// Aligned with InfiniCore and `torch.nn.functional.silu`.
class Silu : public Operator<Silu> {
 public:
  Silu(const Tensor input, Tensor out)
      : ndim_{out.ndim()},
        output_size_{out.numel()},
        input_type_{input.dtype()},
        out_type_{out.dtype()},
        input_shape_{input.shape()},
        out_shape_{out.shape()},
        input_strides_{input.strides()},
        out_strides_{out.strides()},
        is_input_contiguous_{input.IsContiguous()},
        is_out_contiguous_{out.IsContiguous()} {
    assert(input.shape() == out.shape() &&
           "`Silu` requires `input` and `out` to have the same shape");
    assert(input_type_ == out_type_ &&
           "`Silu` requires `input` and `out` to have the same dtype");
    assert((input_type_ == DataType::kFloat16 ||
            input_type_ == DataType::kBFloat16 ||
            input_type_ == DataType::kFloat32 ||
            input_type_ == DataType::kFloat64) &&
           "`Silu` supports float16, bfloat16, float32, and float64 only");
  }

  virtual void operator()(const Tensor input, Tensor out) const = 0;

 protected:
  Tensor::Size ndim_{0};

  Tensor::Size output_size_{0};

  DataType input_type_;

  DataType out_type_;

  Tensor::Shape input_shape_;

  Tensor::Shape out_shape_;

  Tensor::Strides input_strides_;

  Tensor::Strides out_strides_;

  bool is_input_contiguous_{false};

  bool is_out_contiguous_{false};
};

}  // namespace infini::ops

#endif
