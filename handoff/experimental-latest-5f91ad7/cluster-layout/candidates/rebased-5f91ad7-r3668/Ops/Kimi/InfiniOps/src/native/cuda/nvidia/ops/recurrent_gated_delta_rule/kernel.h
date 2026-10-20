#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <climits>

#include "base/recurrent_gated_delta_rule.h"
#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "recurrent_gated_delta_rule_legacy_kernel.cuh"
namespace infini::ops {
namespace recurrent_native_detail {
struct Layout {
  ptrdiff_t s[10][4]{};
  bool initial_i64, final_i64, norm, indexed;
  size_t pool, Hk, ratio;
};
template <typename Data, typename Gate, size_t D>
__global__ void kernel(const Data* q, const Data* k, const Data* v,
                       const Gate* g, const Gate* beta, Data* state,
                       const void* initial_indices, const void* final_indices,
                       Data* out, Data* final_state, Layout p) {
  extern __shared__ float shared[];
  recurrentGatedDeltaRuleIndexedPoolWarpKernel<Data, Gate, float, D, D, 8>(
      out, state, final_state, q, k, v, g, beta, initial_indices, final_indices,
      p.initial_i64, p.final_i64, p.norm, p.indexed, p.pool, p.Hk, p.ratio,
      p.s[8][0], p.s[8][1], p.s[8][2], p.s[5][0], p.s[5][1], p.s[5][2],
      p.s[5][3], p.s[9][0], p.s[9][1], p.s[9][2], p.s[9][3], p.s[0][0],
      p.s[0][1], p.s[0][2], p.s[1][0], p.s[1][1], p.s[1][2], p.s[2][0],
      p.s[2][1], p.s[2][2], p.s[3][0], p.s[3][1], p.s[3][2], p.s[4][0],
      p.s[4][1], p.s[4][2], shared);
}
}  // namespace recurrent_native_detail
template <>
class Operator<RecurrentGatedDeltaRule, Device::Type::kNvidia>
    : public RecurrentGatedDeltaRule {
 public:
  Operator(const Tensor q, const Tensor k, const Tensor v, const Tensor g,
           const Tensor beta, Tensor initial_state,
           const std::optional<Tensor> initial_state_indices,
           const std::optional<Tensor> final_state_indices, Tensor out,
           std::optional<Tensor> final_state, bool use_qk_l2norm)
      : RecurrentGatedDeltaRule(q, k, v, g, beta, initial_state,
                                initial_state_indices, final_state_indices, out,
                                final_state, use_qk_l2norm) {
    if (device_.type() != Device::Type::kNvidia)
      throw std::invalid_argument("recurrent delta requires NVIDIA");
    if (q.shape()[1] != 1 || v.shape()[3] != D_ || (D_ != 64 && D_ != 128))
      throw std::invalid_argument(
          "NVIDIA recurrent delta kernels require T1 and Dk=Dv64/128");
    CheckDevice();
    cudaDeviceProp property{};
    if (cudaGetDeviceProperties(&property, device_.index()) != cudaSuccess ||
        property.maxThreadsPerBlock < 256)
      throw std::runtime_error(
          "recurrent delta requires 256-thread CUDA blocks");
    if (B_ >= static_cast<size_t>(INT_MAX) || Hv_ > 65535 ||
        Hk_ > static_cast<size_t>(INT_MAX))
      throw std::invalid_argument("recurrent delta launch extent unsupported");
    p_.pool = pool_;
    p_.Hk = Hk_;
    p_.ratio = Hv_ / Hk_;
    p_.norm = norm_;
    p_.indexed = metadata_[6].has_value() || metadata_[7].has_value();
    p_.initial_i64 = metadata_[6] && metadata_[6]->dtype() == DataType::kInt64;
    p_.final_i64 = metadata_[7] && metadata_[7]->dtype() == DataType::kInt64;
    for (size_t i = 0; i < metadata_.size(); ++i)
      if (metadata_[i])
        for (size_t j = 0; j < metadata_[i]->strides().size(); ++j)
          p_.s[i][j] = metadata_[i]->strides()[j];
  }
  void operator()(const Tensor q, const Tensor k, const Tensor v,
                  const Tensor g, const Tensor beta, Tensor initial_state,
                  const std::optional<Tensor> initial_state_indices,
                  const std::optional<Tensor> final_state_indices, Tensor out,
                  std::optional<Tensor> final_state,
                  bool use_qk_l2norm) const override {
    const Views views{q,
                      k,
                      v,
                      g,
                      beta,
                      initial_state,
                      initial_state_indices,
                      final_state_indices,
                      out,
                      final_state};
    CheckInvocation(views, use_qk_l2norm);
    CheckDevice();
    if (!B_) return;
    switch (data_type_) {
      case DataType::kFloat16:
        gate<half>(views);
        break;
      case DataType::kBFloat16:
        gate<__nv_bfloat16>(views);
        break;
      case DataType::kFloat32:
        gate<float>(views);
        break;
      default:
        throw std::invalid_argument("recurrent delta data dtype unsupported");
    }
    if (cudaPeekAtLastError() != cudaSuccess)
      throw std::runtime_error("recurrent delta CUDA launch failed");
  }

 private:
  recurrent_native_detail::Layout p_{};
  void CheckDevice() const {
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess)
      throw std::runtime_error("recurrent delta device query failed");
    if (device != device_.index())
      throw std::invalid_argument("recurrent delta owner device mismatch");
  }
  template <typename Data>
  void gate(const Views& v) const {
    switch (gate_type_) {
      case DataType::kFloat16:
        dimension<Data, half>(v);
        break;
      case DataType::kBFloat16:
        dimension<Data, __nv_bfloat16>(v);
        break;
      case DataType::kFloat32:
        dimension<Data, float>(v);
        break;
      default:
        throw std::invalid_argument("recurrent delta gate dtype unsupported");
    }
  }
  template <typename Data, typename Gate>
  void dimension(const Views& v) const {
    if (D_ == 64)
      launch<Data, Gate, 64>(v);
    else
      launch<Data, Gate, 128>(v);
  }
  template <typename Data, typename Gate, size_t D>
  void launch(const Views& v) const {
    auto ptr = [&](size_t i) -> void* {
      return v[i] ? const_cast<void*>(v[i]->data()) : nullptr;
    };
    dim3 grid(static_cast<unsigned int>(B_), static_cast<unsigned int>(Hv_),
              D / 8);
    const auto stream = static_cast<cudaStream_t>(stream_ ? stream_ : nullptr);
    recurrent_native_detail::kernel<Data, Gate, D>
        <<<grid, 256, (D * 3 + 256 + 1) * sizeof(float), stream>>>(
            static_cast<const Data*>(ptr(0)), static_cast<const Data*>(ptr(1)),
            static_cast<const Data*>(ptr(2)), static_cast<const Gate*>(ptr(3)),
            static_cast<const Gate*>(ptr(4)), static_cast<Data*>(ptr(5)),
            ptr(6), ptr(7), static_cast<Data*>(ptr(8)),
            static_cast<Data*>(ptr(9)), p_);
  }
};
}  // namespace infini::ops
