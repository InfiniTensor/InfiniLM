#ifndef INFINI_OPS_BASE_SCALED_SOFTMAX_INFINILM_H_
#define INFINI_OPS_BASE_SCALED_SOFTMAX_INFINILM_H_

#include <cassert>
#include <cmath>
#include <cstddef>

#include "data_type.h"
#include "operator.h"
#include "tensor.h"

namespace infini::ops {

/// \deprecated Migrate to an open-source-aligned operator when available. This
/// interface will be removed in a future release.
class [[deprecated(
    "Migrate to an open-source-aligned operator when available.")]]
ScaledSoftmaxInfinilm : public Operator<ScaledSoftmaxInfinilm> {
 public:
  ScaledSoftmaxInfinilm(const Tensor input, double scale, Tensor out)
      : scale_{scale},
        batch_size_{input.size(0)},
        vocab_size_{input.size(1)},
        dtype_{input.dtype()},
        input_strides_{input.strides()},
        out_strides_{out.strides()} {
    assert(input.ndim() == 2 &&
           "`ScaledSoftmaxInfinilm` currently supports 2D `[batch, vocab]` "
           "input");
    assert(input.shape() == out.shape() &&
           "`ScaledSoftmaxInfinilm` requires `input` and `out` to have the "
           "same shape");
    assert(input.dtype() == out.dtype() &&
           "`ScaledSoftmaxInfinilm` requires `input` and `out` to have the "
           "same dtype");
    assert((dtype_ == DataType::kFloat16 || dtype_ == DataType::kBFloat16 ||
            dtype_ == DataType::kFloat32 || dtype_ == DataType::kFloat64) &&
           "`ScaledSoftmaxInfinilm` requires a floating point dtype");
    assert(std::isfinite(scale_) &&
           "`ScaledSoftmaxInfinilm` requires a finite `scale`");
  }

  virtual void operator()(const Tensor input, double scale,
                          Tensor out) const = 0;

 protected:
  double scale_{1.0};

  Tensor::Size batch_size_{0};

  Tensor::Size vocab_size_{0};

  DataType dtype_;

  Tensor::Strides input_strides_;

  Tensor::Strides out_strides_;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_SCALED_SOFTMAX_INFINILM_H_
