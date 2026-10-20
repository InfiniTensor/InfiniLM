#pragma once
#include "base/deepseek_moe.h"
#include "native/cuda/nvidia/ops/gemm/handle_owner.h"
namespace infini::ops {
template <> class Operator<DeepseekMoe, Device::Type::kNvidia, 0> : public DeepseekMoe {
 public:
  Operator(Tensor output, const Tensor hidden, const Tensor ids, const Tensor weights, const std::vector<Tensor> gate, const std::vector<Tensor> up, const std::vector<Tensor> down, size_t intermediate, size_t num_experts);
  size_t workspace_size_in_bytes() const override;
  bool requires_workspace() const override { return true; }
 private:
  std::shared_ptr<NvidiaGemmHandle> blas_;
 public:
  static bool supports(Tensor output, const Tensor hidden, const Tensor ids, const Tensor weights, const std::vector<Tensor> gate, const std::vector<Tensor> up, const std::vector<Tensor> down, size_t intermediate, size_t num_experts) { return DeepseekMoe::supports(output, hidden, ids, weights, gate, up, down, intermediate, num_experts) && SupportsLayout(hidden, gate, up, down, intermediate); }
  void operator()(Tensor output, const Tensor hidden, const Tensor ids, const Tensor weights, const std::vector<Tensor> gate, const std::vector<Tensor> up, const std::vector<Tensor> down, size_t intermediate, size_t num_experts) const override;
 private:
  static bool SupportsLayout(const Tensor hidden, const std::vector<Tensor>& gate, const std::vector<Tensor>& up, const std::vector<Tensor>& down, size_t intermediate) {
    if(hidden.dtype()!=DataType::kBFloat16) return true;
    const auto align_h=hidden.size(1)%8==0 ? 16u : (hidden.size(1)%2==0 ? 4u : 2u);
    const auto align_m=intermediate%8==0 ? 16u : (intermediate%2==0 ? 4u : 2u);
    if(reinterpret_cast<uintptr_t>(hidden.data())%align_h) return false;
    for(size_t e=0;e<gate.size();++e) if(reinterpret_cast<uintptr_t>(gate[e].data())%align_h || reinterpret_cast<uintptr_t>(up[e].data())%align_h || reinterpret_cast<uintptr_t>(down[e].data())%align_m) return false;
    return true;
  }
};
}
