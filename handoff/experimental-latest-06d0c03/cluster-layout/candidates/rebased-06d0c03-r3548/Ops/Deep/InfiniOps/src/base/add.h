#ifndef INFINI_OPS_BASE_ADD_H_
#define INFINI_OPS_BASE_ADD_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "operator.h"

namespace infini::ops {

class Add : public Operator<Add> {
 public:
  Add(const Tensor input, const Tensor other, const double alpha, Tensor out)
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
           "the output of `Add` should NOT have broadcasted dim!");
    // TODO(lzm): support mix-precision later using the generic elementwise
    // framework.
    assert(input_type_ == other_type_ && other_type_ == out_type_ &&
           "operator `Add` requires all input and output Tensors to have the "
           "same dtype");
    if (input_type_ != DataType::kFloat16 &&
        input_type_ != DataType::kBFloat16 &&
        input_type_ != DataType::kFloat32 &&
        input_type_ != DataType::kFloat64) {
      assert(alpha == static_cast<int64_t>(alpha) &&
             "operator `Add` requires integral `alpha` for integer tensors");
    }
    ValidateBroadcast(input, other, out);
  }

  Add(const Tensor input, const Tensor other, Tensor out)
      : Add{input, other, 1.0, out} {}

  virtual void operator()(const Tensor input, const Tensor other,
                          const double alpha, Tensor out) const = 0;

  void operator()(const Tensor input, const Tensor other, Tensor out) const {
    (*this)(input, other, 1.0, out);
  }

  template <typename TensorLike>
  static auto MakeReturnValue(const TensorLike& input,
                              const TensorLike& other) {
    return MakeReturnValue(input, other, 1.0);
  }

  template <typename TensorLike>
  static auto MakeReturnValue(const TensorLike& input, const TensorLike& other,
                              const double /*alpha*/) {
    auto input_shape = input.shape();
    auto other_shape = other.shape();
    auto ndim = std::max(input_shape.size(), other_shape.size());
    typename TensorLike::Shape out_shape(ndim, 1);

    for (std::size_t i = 0; i < ndim; ++i) {
      auto input_dim = i < ndim - input_shape.size()
                           ? 1
                           : input_shape[i + input_shape.size() - ndim];
      auto other_dim = i < ndim - other_shape.size()
                           ? 1
                           : other_shape[i + other_shape.size() - ndim];
      assert((input_dim == other_dim || input_dim == 1 || other_dim == 1) &&
             "operator `Add` requires broadcast-compatible input shapes");
      out_shape[i] = std::max(input_dim, other_dim);
    }

    return TensorLike::Empty(out_shape, input.dtype(), input.device());
  }

 protected:
  static Tensor::Strides BroadcastStrides(const Tensor input,
                                          const Tensor out) {
    assert(input.ndim() <= out.ndim() &&
           "operator `Add` input rank must not exceed output rank");
    Tensor::Strides strides(out.ndim(), 0);
    auto offset = out.ndim() - input.ndim();

    for (Tensor::Size i = 0; i < input.ndim(); ++i) {
      auto out_dim = i + offset;
      assert((input.size(i) == 1 || input.size(i) == out.size(out_dim)) &&
             "operator `Add` input shape is not broadcast-compatible with "
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
      assert(out.size(i) == std::max(input_dim, other_dim) &&
             "operator `Add` output shape must equal the broadcasted input "
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
