#ifndef INFINI_OPS_BASE_SILU_AND_MUL_INFINILM_H_
#define INFINI_OPS_BASE_SILU_AND_MUL_INFINILM_H_

#include <cassert>

#include "operator.h"

namespace infini::ops {

/// \deprecated Use `SiluAndMul`. This interface will be removed in a future
/// release.
class [[deprecated("Use `SiluAndMul` instead.")]] SiluAndMulInfinilm
    : public Operator<SiluAndMulInfinilm> {
 public:
  SiluAndMulInfinilm(const Tensor input, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        output_size_{out.numel()},
        ndim_{out.ndim()},
        hidden_size_{out.size(out.ndim() - 1)},
        row_count_{out.numel() / hidden_size_},
        device_index_{out.device().index()} {
    assert(input.ndim() == out.ndim() &&
           "`SiluAndMulInfinilm` input and output ranks must match");
    assert(input_type_ == out_type_ &&
           "`SiluAndMulInfinilm` input and output dtypes must match");
    assert(
        input.size(input.ndim() - 1) == 2 * hidden_size_ &&
        "`SiluAndMulInfinilm` input last dimension must be twice output last "
        "dimension");
    for (Tensor::Size i = 0; i + 1 < ndim_; ++i) {
      assert(input.size(i) == out.size(i) &&
             "`SiluAndMulInfinilm` leading dimensions must match");
    }
    assert(input.IsContiguous() && out.IsContiguous() &&
           "`SiluAndMulInfinilm` only supports contiguous tensors");
    assert(!out.HasBroadcastDim() &&
           "`SiluAndMulInfinilm` output must not have broadcasted dimensions");
  }

  virtual void operator()(const Tensor input, Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  Tensor::Size output_size_{0};

  Tensor::Size ndim_{0};

  Tensor::Size hidden_size_{0};

  Tensor::Size row_count_{0};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
