#ifndef INFINI_OPS_BASE_FUSED_ADD_RMS_NORM_H_
#define INFINI_OPS_BASE_FUSED_ADD_RMS_NORM_H_

#include <cstddef>
#include <optional>

#include "operator.h"
#include "tensor.h"

namespace infini::ops {

class FusedAddRmsNorm : public Operator<FusedAddRmsNorm> {
 public:
  FusedAddRmsNorm(Tensor input, Tensor residual,
                  const std::optional<Tensor> weight, float epsilon)
      : input_strides_{input.strides()},
        residual_strides_{residual.strides()},
        epsilon_{epsilon},
        dim_{input.size(-1)},
        num_tokens_{dim_ == 0 ? 0 : input.numel() / dim_} {
    assert(input.ndim() >= 2 &&
           "`FusedAddRmsNorm` requires `input` to have at least 2 dimensions");
    assert(dim_ > 0 &&
           "`FusedAddRmsNorm` requires a non-empty normalized dimension");
    assert(input.shape() == residual.shape() &&
           "`FusedAddRmsNorm` requires `input` and `residual` to have the same "
           "shape");
    assert(input.dtype() == residual.dtype() &&
           "`FusedAddRmsNorm` requires `input` and `residual` to have the same "
           "dtype");
    assert(input.stride(-1) == 1 &&
           "`FusedAddRmsNorm` requires the last dimension of `input` to be "
           "contiguous");
    assert(residual.stride(-1) == 1 &&
           "`FusedAddRmsNorm` requires the last dimension of `residual` to be "
           "contiguous");

    for (Tensor::Size i = 0; i + 2 < input.ndim(); ++i) {
      assert(input.stride(i) == input.size(i + 1) * input.stride(i + 1) &&
             "`FusedAddRmsNorm` requires `input` rows to have a uniform "
             "stride");
      assert(residual.stride(i) ==
                 residual.size(i + 1) * residual.stride(i + 1) &&
             "`FusedAddRmsNorm` requires `residual` rows to have a uniform "
             "stride");
    }

    if (weight.has_value()) {
      assert(weight->ndim() == 1 && weight->size(0) == dim_ &&
             "`FusedAddRmsNorm` requires 1D `weight` with size equal to the "
             "normalized dimension");
      assert(weight->dtype() == input.dtype() &&
             "`FusedAddRmsNorm` requires `input` and `weight` to have the "
             "same dtype");
      assert(weight->stride(0) == 1 &&
             "`FusedAddRmsNorm` requires `weight` to be contiguous");
    }
  }

  virtual void operator()(Tensor input, Tensor residual,
                          const std::optional<Tensor> weight,
                          float epsilon) const = 0;

 protected:
  Tensor::Strides input_strides_;

  Tensor::Strides residual_strides_;

  float epsilon_{};

  Tensor::Size dim_{0};

  Tensor::Size num_tokens_{0};
};

}  // namespace infini::ops

#endif
