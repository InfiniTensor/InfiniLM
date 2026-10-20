#ifndef INFINI_OPS_BASE_SILU_AND_MUL_H_
#define INFINI_OPS_BASE_SILU_AND_MUL_H_

#include "data_type.h"
#include "operator.h"

namespace infini::ops {

class SiluAndMul : public Operator<SiluAndMul> {
 public:
  SiluAndMul(const Tensor input, Tensor out)
      : ndim_{out.ndim()},
        output_size_{out.numel()},
        input_type_{input.dtype()},
        out_type_{out.dtype()},
        input_shape_{input.shape()},
        out_shape_{out.shape()},
        input_strides_{input.strides()},
        out_strides_{out.strides()},
        hidden_size_{out.size(-1)},
        is_input_contiguous_{input.IsContiguous()},
        is_out_contiguous_{out.IsContiguous()} {
    assert(input.ndim() == out.ndim() &&
           "`SiluAndMul` requires input and output ranks to match");
    assert(input_type_ == out_type_ &&
           "`SiluAndMul` requires input and output dtypes to match");
    assert(input.size(-1) == 2 * hidden_size_ &&
           "`SiluAndMul` requires input last dimension to be twice output "
           "last dimension");
    for (Tensor::Size i = 0; i + 1 < ndim_; ++i) {
      assert(input.size(i) == out.size(i) &&
             "`SiluAndMul` requires matching leading dimensions");
    }
  }

  virtual void operator()(const Tensor input, Tensor out) const = 0;

  template <typename TensorLike>
  static auto MakeReturnValue(const TensorLike& input) {
    typename TensorLike::Shape out_shape{input.shape()};
    assert(!out_shape.empty() && out_shape.back() % 2 == 0 &&
           "`SiluAndMul` requires an even input last dimension");
    out_shape.back() /= 2;

    return TensorLike::Empty(out_shape, input.dtype(), input.device());
  }

 protected:
  Tensor::Size ndim_{0};

  Tensor::Size output_size_{0};

  DataType input_type_;

  DataType out_type_;

  Tensor::Shape input_shape_;

  Tensor::Shape out_shape_;

  Tensor::Strides input_strides_;

  Tensor::Strides out_strides_;

  Tensor::Size hidden_size_{0};

  bool is_input_contiguous_{false};

  bool is_out_contiguous_{false};
};

}  // namespace infini::ops

#endif
