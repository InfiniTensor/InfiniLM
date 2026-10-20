#pragma once
#include "base/moe_fused_dense.h"
namespace infini::ops {
template <> class Operator<MoeFusedDense, Device::Type::kNvidia, 0> : public MoeFusedDense {
 public:
  Operator(Tensor output, const Tensor hidden, const Tensor w13, const Tensor w2, const Tensor weights, const Tensor ids, const Tensor sorted, const Tensor experts, const Tensor total, size_t block_size, const std::optional<Tensor> bias1 = std::nullopt, const std::optional<Tensor> bias2 = std::nullopt);
  size_t workspace_size_in_bytes() const override;
  bool requires_workspace() const override { return true; }
  static bool supports(Tensor output, const Tensor hidden, const Tensor w13, const Tensor w2, const Tensor weights, const Tensor ids, const Tensor sorted, const Tensor experts, const Tensor total, size_t block_size, const std::optional<Tensor> bias1 = std::nullopt, const std::optional<Tensor> bias2 = std::nullopt) { return MoeFusedDense::supports(output, hidden, w13, w2, weights, ids, sorted, experts, total, block_size, bias1, bias2) && SupportsLayout(hidden,w13,w2); }
  void operator()(Tensor output, const Tensor hidden, const Tensor w13, const Tensor w2, const Tensor weights, const Tensor ids, const Tensor sorted, const Tensor experts, const Tensor total, size_t block_size, const std::optional<Tensor> bias1 = std::nullopt, const std::optional<Tensor> bias2 = std::nullopt) const override;
 private:
  static bool SupportsLayout(const Tensor hidden, const Tensor w13, const Tensor w2) {
    return w13.size(0)<=1024 && hidden.size(1)%8==0 && w2.size(2)%8==0 &&
        reinterpret_cast<uintptr_t>(w13.data())%16==0 && reinterpret_cast<uintptr_t>(w2.data())%16==0 &&
        (hidden.size(0)!=1 || reinterpret_cast<uintptr_t>(hidden.data())%16==0);
  }
};
}
