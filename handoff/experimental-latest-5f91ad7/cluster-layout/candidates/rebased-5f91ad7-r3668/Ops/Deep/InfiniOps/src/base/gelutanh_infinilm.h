#ifndef INFINI_OPS_BASE_GELUTANH_INFINILM_H_
#define INFINI_OPS_BASE_GELUTANH_INFINILM_H_

#include <cassert>

#include "operator.h"

namespace infini::ops {

/// \deprecated Use `Gelu`. This interface will be removed in a future release.
class [[deprecated("Use `Gelu` instead.")]] GelutanhInfinilm
    : public Operator<GelutanhInfinilm> {
 public:
  GelutanhInfinilm(const Tensor input, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        output_size_{out.numel()},
        ndim_{out.ndim()},
        is_input_contiguous_{input.IsContiguous()},
        is_out_contiguous_{out.IsContiguous()},
        device_index_{out.device().index()} {
    assert(input_shape_ == out_shape_ &&
           "`GelutanhInfinilm` input and output shapes must match");
    assert(input_type_ == out_type_ &&
           "`GelutanhInfinilm` input and output dtypes must match");
    assert(!out.HasBroadcastDim() &&
           "`GelutanhInfinilm` output must not have broadcasted dimensions");
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

  bool is_input_contiguous_{false};

  bool is_out_contiguous_{false};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
