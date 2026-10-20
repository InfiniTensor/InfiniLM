#ifndef INFINI_OPS_BASE_COPY_H_
#define INFINI_OPS_BASE_COPY_H_

#include <cassert>

#include "operator.h"

namespace infini::ops {

class Copy : public Operator<Copy> {
 public:
  Copy(const Tensor src, const bool non_blocking, Tensor out)
      : input_shape_{out.shape()},
        input_strides_{BroadcastStrides(src, out)},
        input_type_{src.dtype()},
        non_blocking_{non_blocking},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        output_size_{out.numel()},
        ndim_{out.ndim()},
        is_input_contiguous_{src.shape() == out.shape() && src.IsContiguous()},
        is_out_contiguous_{out.IsContiguous()} {
    assert(input_type_ == out_type_ &&
           "`Copy` currently requires input and output dtypes to match");
    assert(src.device() == out.device() &&
           "`Copy` currently requires input and output on the same device");
    assert(!out.HasBroadcastDim() &&
           "`Copy` output must not have broadcasted dimensions");
  }

  virtual void operator()(const Tensor src, const bool non_blocking,
                          Tensor out) const = 0;

 protected:
  static bool SameView(const Tensor& src, const Tensor& out) {
    return src.data() == out.data() && src.shape() == out.shape() &&
           src.strides() == out.strides();
  }

  static Tensor::Strides BroadcastStrides(const Tensor src, const Tensor out) {
    assert(src.ndim() <= out.ndim() &&
           "`Copy` input rank must not exceed output rank");
    Tensor::Strides strides(out.ndim(), 0);
    auto offset = out.ndim() - src.ndim();

    for (Tensor::Size i = 0; i < src.ndim(); ++i) {
      auto out_dim = i + offset;
      assert((src.size(i) == 1 || src.size(i) == out.size(out_dim)) &&
             "`Copy` input shape must be broadcastable to output shape");
      strides[out_dim] = src.size(i) == 1 ? 0 : src.stride(i);
    }

    return strides;
  }

  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  bool non_blocking_{false};

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  Tensor::Size output_size_{0};

  Tensor::Size ndim_{0};

  bool is_input_contiguous_{false};

  bool is_out_contiguous_{false};
};

}  // namespace infini::ops

#endif
