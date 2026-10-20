#pragma once
#include <array>
#include <functional>
#include <stdexcept>

#include "data_type.h"
#include "operator.h"

namespace infini::ops {
class FusedGatedDeltaNetGating : public Operator<FusedGatedDeltaNetGating> {
 public:
  FusedGatedDeltaNetGating(Tensor A_log, Tensor a, Tensor b, Tensor dt_bias,
                           float beta, float threshold, Tensor g,
                           Tensor beta_output)
      : beta_(beta),
        threshold_(threshold),
        metadata_{A_log, a, b, dt_bias, g, beta_output} {
    const auto& tensors = metadata_;
    if (a.ndim() != 3 || b.shape() != a.shape() || g.shape() != a.shape() ||
        beta_output.shape() != a.shape() || A_log.ndim() != 1 ||
        dt_bias.ndim() != 1 || A_log.numel() != a.shape()[2] ||
        dt_bias.numel() != a.shape()[2])
      throw std::invalid_argument("gating shape mismatch");
    if ((a.dtype() != DataType::kFloat32 && a.dtype() != DataType::kFloat16 &&
         a.dtype() != DataType::kBFloat16) ||
        b.dtype() != a.dtype() || A_log.dtype() != a.dtype() ||
        dt_bias.dtype() != a.dtype() || g.dtype() != DataType::kFloat32 ||
        beta_output.dtype() != DataType::kFloat32)
      throw std::invalid_argument("gating dtype mismatch");
    device_ = a.device();
    total_ = a.numel();
    seq_ = a.shape()[1];
    hidden_ = a.shape()[2];
    for (size_t i = 0; i < tensors.size(); ++i) {
      if (tensors[i].device() != device_ ||
          (i >= 4 && tensors[i].HasBroadcastDim()))
        throw std::invalid_argument("gating device/output layout mismatch");
    }
  }
  virtual void operator()(Tensor A_log, Tensor a, Tensor b, Tensor dt_bias,
                          float beta, float threshold, Tensor g,
                          Tensor beta_output) const = 0;

 protected:
  Device device_;
  size_t total_, seq_, hidden_;
  float beta_, threshold_;
  const std::array<Tensor, 6> metadata_;
};
}  // namespace infini::ops
