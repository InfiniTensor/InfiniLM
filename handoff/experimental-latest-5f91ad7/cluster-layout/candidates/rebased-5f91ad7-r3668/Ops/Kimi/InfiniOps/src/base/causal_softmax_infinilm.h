#ifndef INFINI_OPS_BASE_CAUSAL_SOFTMAX_INFINILM_H_
#define INFINI_OPS_BASE_CAUSAL_SOFTMAX_INFINILM_H_

#include <cassert>
#include <cstddef>

#include "operator.h"
#include "tensor.h"

namespace infini::ops {

/// \deprecated Migrate to an open-source-aligned operator when available. This
/// interface will be removed in a future release.
class [[deprecated(
    "Migrate to an open-source-aligned operator when available.")]]
CausalSoftmaxInfinilm : public Operator<CausalSoftmaxInfinilm> {
 public:
  CausalSoftmaxInfinilm(const Tensor input, Tensor out)
      : dtype_{input.dtype()},
        ndim_{out.ndim()},
        batch_size_{ndim_ == 2 ? 1 : out.size(-3)},
        seq_len_{out.size(-2)},
        total_seq_len_{out.size(-1)},
        input_strides_{input.strides()},
        out_strides_{out.strides()} {
    assert(input.shape() == out.shape() &&
           "`CausalSoftmaxInfinilm` requires `input` and `out` same shape");
    assert(input.dtype() == out.dtype() &&
           "`CausalSoftmaxInfinilm` requires `input` and `out` same dtype");
    assert((ndim_ == 2 || ndim_ == 3) &&
           "`CausalSoftmaxInfinilm` requires 2D or 3D tensor");
    assert(seq_len_ <= total_seq_len_ &&
           "`CausalSoftmaxInfinilm` requires shape[-2] <= shape[-1]");
  }

  virtual void operator()(const Tensor input, Tensor out) const = 0;

 protected:
  const DataType dtype_;

  Tensor::Size ndim_{0};

  Tensor::Size batch_size_{0};

  Tensor::Size seq_len_{0};

  Tensor::Size total_seq_len_{0};

  Tensor::Strides input_strides_;

  Tensor::Strides out_strides_;
};

}  // namespace infini::ops

#endif
