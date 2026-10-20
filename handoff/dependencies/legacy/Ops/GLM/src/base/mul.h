#ifndef INFINI_OPS_BASE_MUL_H_
#define INFINI_OPS_BASE_MUL_H_

#include "operator.h"

namespace infini::ops {

class Mul : public Operator<Mul> {
 public:
  Mul(const Tensor input, const Tensor other, Tensor out)
      : ndim_{out.ndim()},
        output_size_{out.numel()},
        input_type_{input.dtype()},
        other_type_{other.dtype()},
        out_type_{out.dtype()},
        input_shape_{out.shape()},
        other_shape_{out.shape()},
        out_shape_{out.shape()},
        input_strides_{BroadcastStrides(input, out)},
        other_strides_{BroadcastStrides(other, out)},
        out_strides_{out.strides()},
        is_input_contiguous_{input.shape() == out.shape() &&
                             input.IsContiguous()},
        is_other_contiguous_{other.shape() == out.shape() &&
                             other.IsContiguous()},
        is_out_contiguous_{out.IsContiguous()} {
    assert(!out.HasBroadcastDim() &&
           "the output of `Mul` should NOT have broadcasted dim!");
    assert(input_type_ == other_type_ && other_type_ == out_type_ &&
           "operator `Mul` requires all input and output tensors to have the "
           "same dtype");
    ValidateBroadcast(input, other, out);
  }

  virtual void operator()(const Tensor input, const Tensor other,
                          Tensor out) const = 0;

 protected:
  static Tensor::Strides BroadcastStrides(const Tensor input,
                                          const Tensor out) {
    assert(input.ndim() <= out.ndim() &&
           "operator `Mul` input rank must not exceed output rank");
    Tensor::Strides strides(out.ndim(), 0);
    auto offset = out.ndim() - input.ndim();

    for (Tensor::Size i = 0; i < input.ndim(); ++i) {
      auto out_dim = i + offset;
      assert((input.size(i) == 1 || input.size(i) == out.size(out_dim)) &&
             "operator `Mul` input shape is not broadcast-compatible with "
             "output shape");
      strides[out_dim] = input.size(i) == 1 ? 0 : input.stride(i);
    }

    return strides;
  }

  static void ValidateBroadcast(const Tensor input, const Tensor other,
                                const Tensor out) {
    for (Tensor::Size i = 0; i < out.ndim(); ++i) {
      auto input_dim = i < out.ndim() - input.ndim()
                           ? 1
                           : input.size(i + input.ndim() - out.ndim());
      auto other_dim = i < out.ndim() - other.ndim()
                           ? 1
                           : other.size(i + other.ndim() - out.ndim());
      [[maybe_unused]] auto broadcast_dim =
          input_dim == 1 ? other_dim : input_dim;
      assert(out.size(i) == broadcast_dim &&
             "operator `Mul` output shape must equal the broadcasted input "
             "shape");
    }
  }

  Tensor::Size ndim_{0};

  Tensor::Size output_size_{0};

  const DataType input_type_;

  const DataType other_type_;

  const DataType out_type_;

  Tensor::Shape input_shape_;

  Tensor::Shape other_shape_;

  Tensor::Shape out_shape_;

  Tensor::Strides input_strides_;

  Tensor::Strides other_strides_;

  Tensor::Strides out_strides_;

  bool is_input_contiguous_{false};

  bool is_other_contiguous_{false};

  bool is_out_contiguous_{false};
};

}  // namespace infini::ops

#endif
