#pragma once
#include "base/chunk_gated_delta_rule.h"
#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "chunk_gated_delta_rule_legacy_kernel.cuh"
#include <climits>
namespace infini::ops {
namespace chunk_native_detail {
struct Layout {
  ptrdiff_t s[11][4]{};
  bool cu_i64, initial_i64, final_i64, norm, indexed, packed;
  size_t T, chunk, pool, Hk, ratio;
};
template <typename Data, typename Gate, size_t D>
__global__ void kernel(float *workspace, const Data *q, const Data *k, const Data *v,
    const Gate *g, const Gate *beta, Data *state, const void *cu, const void *initial_indices,
    const void *final_indices, Data *out, Data *final_state, Layout p) {
  chunkGatedDeltaRuleRecurrentKernel<Data, Gate, float, D, D, D>(
      workspace, out, state, final_state, q, k, v, g, beta, cu, initial_indices, final_indices,
      p.cu_i64, p.initial_i64, p.final_i64, p.norm, p.packed, p.indexed, p.T, p.chunk, p.pool, p.Hk, p.ratio,
      p.s[9][0], p.s[9][1], p.s[9][2], p.s[5][0], p.s[5][1], p.s[5][2], p.s[5][3],
      p.s[10][0], p.s[10][1], p.s[10][2], p.s[10][3], p.s[0][0], p.s[0][1], p.s[0][2],
      p.s[1][0], p.s[1][1], p.s[1][2], p.s[2][0], p.s[2][1], p.s[2][2],
      p.s[3][0], p.s[3][1], p.s[3][2], p.s[4][0], p.s[4][1], p.s[4][2]);
}
}
template <>
class Operator<ChunkGatedDeltaRule, Device::Type::kNvidia> : public ChunkGatedDeltaRule {
 public:
  Operator(const Tensor q, const Tensor k, const Tensor v, const Tensor g, const Tensor beta,
      Tensor initial_state, const std::optional<Tensor> cu_seqlens,
      const std::optional<Tensor> initial_state_indices, const std::optional<Tensor> final_state_indices,
      Tensor out, std::optional<Tensor> final_state, bool use_qk_l2norm, size_t chunk_size)
      : ChunkGatedDeltaRule(q, k, v, g, beta, initial_state, cu_seqlens,
                           initial_state_indices, final_state_indices, out, final_state, use_qk_l2norm, chunk_size) {
    if (device_.type() != Device::Type::kNvidia) throw std::invalid_argument("chunk delta requires NVIDIA");
    CheckDevice();
    cudaDeviceProp property{};
    if (cudaGetDeviceProperties(&property, device_.index()) != cudaSuccess || property.maxThreadsPerBlock < int(D_))
      throw std::runtime_error("chunk delta CUDA block size unsupported");
    if (B_ >= static_cast<size_t>(INT_MAX) || Hv_ > 65535 || Hk_ > static_cast<size_t>(INT_MAX))
      throw std::invalid_argument("chunk delta launch extent unsupported");
    const auto maximum = std::numeric_limits<size_t>::max();
    const size_t per_head = D_ * D_ * sizeof(float);
    if (B_ > maximum / Hv_ || B_ * Hv_ > maximum / per_head)
      throw std::overflow_error("chunk delta workspace size overflows");
    bytes_ = B_ * Hv_ * per_head;
    if (bytes_ / sizeof(float) > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()))
      throw std::overflow_error("chunk delta workspace element offset exceeds ptrdiff");
    p_.pool = pool_; p_.Hk = Hk_; p_.ratio = Hv_ / Hk_; p_.norm = norm_; p_.T = T_; p_.chunk = chunk_;
    p_.packed = metadata_[6].present; p_.indexed = metadata_[7].present || metadata_[8].present;
    p_.cu_i64 = metadata_[6].dtype == DataType::kInt64;
    p_.initial_i64 = metadata_[7].dtype == DataType::kInt64;
    p_.final_i64 = metadata_[8].dtype == DataType::kInt64;
    for (size_t i = 0; i < metadata_.size(); ++i)
      for (size_t j = 0; j < metadata_[i].strides.size(); ++j) p_.s[i][j] = metadata_[i].strides[j];
  }
  size_t workspace_size_in_bytes() const override { return bytes_; }
  bool capture_safe() const override { return false; }
  void operator()(const Tensor q, const Tensor k, const Tensor v, const Tensor g, const Tensor beta,
      Tensor initial_state, const std::optional<Tensor> cu_seqlens,
      const std::optional<Tensor> initial_state_indices, const std::optional<Tensor> final_state_indices,
      Tensor out, std::optional<Tensor> final_state, bool use_qk_l2norm, size_t chunk_size) const override {
    const Views views{q, k, v, g, beta, initial_state, cu_seqlens, initial_state_indices, final_state_indices, out, final_state};
    CheckInvocation(views, use_qk_l2norm, chunk_size); CheckDevice();
    if (!B_) return;
    if (!workspace_ || workspace_size_in_bytes_ < bytes_)
      throw std::invalid_argument("chunk delta workspace missing or too small");
    switch (data_type_) {
      case DataType::kFloat16: gate<half>(views); break;
      case DataType::kBFloat16: gate<__nv_bfloat16>(views); break;
      case DataType::kFloat32: gate<float>(views); break;
      default: throw std::invalid_argument("chunk delta data dtype unsupported");
    }
    if (cudaPeekAtLastError() != cudaSuccess) throw std::runtime_error("chunk delta CUDA launch failed");
  }
 private:
  size_t bytes_{0}; chunk_native_detail::Layout p_{};
  void CheckDevice() const {
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) throw std::runtime_error("chunk delta device query failed");
    if (device != device_.index()) throw std::invalid_argument("chunk delta owner device mismatch");
  }
  template <typename Data> void gate(const Views &v) const {
    switch (gate_type_) {
      case DataType::kFloat16: dimension<Data, half>(v); break;
      case DataType::kBFloat16: dimension<Data, __nv_bfloat16>(v); break;
      case DataType::kFloat32: dimension<Data, float>(v); break;
      default: throw std::invalid_argument("chunk delta gate dtype unsupported");
    }
  }
  template <typename Data, typename Gate> void dimension(const Views &v) const {
    if (D_ == 64) launch<Data, Gate, 64>(v); else launch<Data, Gate, 128>(v);
  }
  template <typename Data, typename Gate, size_t D> void launch(const Views &v) const {
    auto ptr = [&](size_t i) -> void * { return v[i] ? const_cast<void *>(v[i]->data()) : nullptr; };
    dim3 grid(static_cast<unsigned int>(B_), static_cast<unsigned int>(Hv_), 1);
    const auto stream = static_cast<cudaStream_t>(stream_ ? stream_ : nullptr);
    chunk_native_detail::kernel<Data, Gate, D><<<grid, D, (D + D + D) * sizeof(float), stream>>>(
        static_cast<float *>(workspace_), static_cast<const Data *>(ptr(0)), static_cast<const Data *>(ptr(1)),
        static_cast<const Data *>(ptr(2)), static_cast<const Gate *>(ptr(3)), static_cast<const Gate *>(ptr(4)),
        static_cast<Data *>(ptr(5)), ptr(6), ptr(7), ptr(8), static_cast<Data *>(ptr(9)), static_cast<Data *>(ptr(10)), p_);
  }
};
}
