#ifndef INFINI_OPS_BASE_ADD_RMS_NORM_H_
#define INFINI_OPS_BASE_ADD_RMS_NORM_H_

#include <cstddef>
#include <optional>

#include "operator.h"
#include "tensor.h"

namespace infini::ops {

// Legacy out-of-place fused add + RMSNorm interface.
/// \deprecated Use `FusedAddRmsNorm`. This interface will be removed in a
/// future release.
class [[deprecated("Use `FusedAddRmsNorm` instead.")]] AddRmsNorm
    : public Operator<AddRmsNorm> {
 public:
  AddRmsNorm(const Tensor input, const Tensor residual, const Tensor weight,
             std::optional<float> eps, Tensor out, Tensor residual_out)
      : input_shape_{input.shape()},
        out_shape_{out.shape()},
        input_strides_{input.strides()},
        residual_strides_{residual.strides()},
        out_strides_{out.strides()},
        residual_out_strides_{residual_out.strides()},
        eps_{eps.value_or(1e-6f)},
        dim_{out.size(-1)},
        ndim_{out.ndim()},
        batch_size_{ndim_ == 2 ? out.size(-2) : out.size(-3)},
        nhead_{ndim_ == 2 ? 1 : out.size(-2)} {
    assert((ndim_ == 2 || ndim_ == 3) &&
           "`AddRmsNorm` supports 2D or 3D tensors only");
    assert(input.shape() == out.shape() &&
           "`AddRmsNorm` requires `input` and `out` to have the same shape");
    assert(input.shape() == residual.shape() &&
           "`AddRmsNorm` requires `input` and `residual` to have the same "
           "shape");
    assert(input.shape() == residual_out.shape() &&
           "`AddRmsNorm` requires `input` and `residual_out` to have the "
           "same shape");
    assert(weight.ndim() == 1 && weight.size(-1) == dim_ &&
           "`AddRmsNorm` requires 1D `weight` with size equal to the "
           "normalized dimension");
    assert(input.dtype() == out.dtype() &&
           "`AddRmsNorm` requires `input` and `out` to have the same dtype");
    assert(input.dtype() == residual.dtype() &&
           "`AddRmsNorm` requires `input` and `residual` to have the same "
           "dtype");
    assert(input.dtype() == residual_out.dtype() &&
           "`AddRmsNorm` requires `input` and `residual_out` to have the same "
           "dtype");
    // The CUDA kernel indexes the normalized dimension with stride 1.
    assert(input.stride(-1) == 1 &&
           "`AddRmsNorm` requires the last dimension of `input` to be "
           "contiguous");
    assert(residual.stride(-1) == 1 &&
           "`AddRmsNorm` requires the last dimension of `residual` to be "
           "contiguous");
    assert(out.stride(-1) == 1 &&
           "`AddRmsNorm` requires the last dimension of `out` to be "
           "contiguous");
    assert(residual_out.stride(-1) == 1 &&
           "`AddRmsNorm` requires the last dimension of `residual_out` to be "
           "contiguous");
    assert(weight.stride(-1) == 1 &&
           "`AddRmsNorm` requires the last dimension of `weight` to be "
           "contiguous");
  }

  virtual void operator()(const Tensor input, const Tensor residual,
                          const Tensor weight, std::optional<float> eps,
                          Tensor out, Tensor residual_out) const = 0;

  virtual void operator()(const Tensor input, const Tensor residual,
                          const Tensor weight, Tensor out,
                          Tensor residual_out) const {
    return operator()(input, residual, weight, std::nullopt, out, residual_out);
  }

 protected:
  Tensor::Shape input_shape_;

  Tensor::Shape out_shape_;

  Tensor::Strides input_strides_;

  Tensor::Strides residual_strides_;

  Tensor::Strides out_strides_;

  Tensor::Strides residual_out_strides_;

  float eps_{1e-6f};

  Tensor::Size dim_{0};

  Tensor::Size ndim_{0};

  Tensor::Size batch_size_{0};

  Tensor::Size nhead_{1};
};

}  // namespace infini::ops

#endif
