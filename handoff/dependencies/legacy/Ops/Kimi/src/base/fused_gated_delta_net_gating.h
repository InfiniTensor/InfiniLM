#pragma once
#include "operator.h"
#include "data_type.h"
#include <array>
#include <stdexcept>

namespace infini::ops {
// Candidate destination: src/base/fused_gated_delta_net_gating.h.
class FusedGatedDeltaNetGating : public Operator<FusedGatedDeltaNetGating> {
 public:
  FusedGatedDeltaNetGating(Tensor A_log, Tensor a, Tensor b, Tensor dt_bias,
                          float beta, float threshold, Tensor g, Tensor beta_output)
      : beta_(beta), threshold_(threshold) {
    std::array<Tensor, 6> tensors{A_log, a, b, dt_bias, g, beta_output};
    if (a.ndim() != 3 || b.shape() != a.shape() || g.shape() != a.shape() ||
        beta_output.shape() != a.shape() || A_log.ndim() != 1 || dt_bias.ndim() != 1 ||
        A_log.numel() != a.shape()[2] || dt_bias.numel() != a.shape()[2])
      throw std::invalid_argument("gating shape mismatch");
    if ((a.dtype() != DataType::kFloat32 && a.dtype() != DataType::kFloat16 &&
         a.dtype() != DataType::kBFloat16) || b.dtype() != a.dtype() ||
        A_log.dtype() != a.dtype() || dt_bias.dtype() != a.dtype() ||
        g.dtype() != DataType::kFloat32 || beta_output.dtype() != DataType::kFloat32)
      throw std::invalid_argument("gating dtype mismatch");
    device_ = a.device();
    total_ = a.numel(); seq_ = a.shape()[1]; hidden_ = a.shape()[2];
    for (size_t i = 0; i < tensors.size(); ++i) {
      if (tensors[i].device() != device_ || (i >= 4 && tensors[i].HasBroadcastDim()))
        throw std::invalid_argument("gating device/output layout mismatch");
      shapes_[i].assign(tensors[i].shape().begin(),tensors[i].shape().end());
      strides_[i].assign(tensors[i].strides().begin(),tensors[i].strides().end());
      types_[i] = tensors[i].dtype();
    }
  }
  virtual void operator()(Tensor A_log, Tensor a, Tensor b, Tensor dt_bias,
                          float beta, float threshold, Tensor g, Tensor beta_output) const = 0;
 protected:
  Device device_;
  size_t total_, seq_, hidden_;
  float beta_, threshold_;
  std::array<Tensor::Shape, 6> shapes_;
  std::array<Tensor::Strides, 6> strides_;
  std::array<DataType, 6> types_;
};
}
