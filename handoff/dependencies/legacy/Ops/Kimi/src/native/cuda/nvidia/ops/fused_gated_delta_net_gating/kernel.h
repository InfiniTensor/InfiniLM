#pragma once
// Candidate destination: src/native/cuda/nvidia/ops/fused_gated_delta_net_gating/kernel.h.
#include "base/fused_gated_delta_net_gating.h"
#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <limits>
#include <cstring>

namespace infini::ops {
namespace gating_detail {
struct Layout {
  ptrdiff_t strides[6][3]{};
  size_t total, seq, hidden;
  float beta, threshold;
};
template <typename T>
__global__ void kernel(const T *A, const T *a, const T *b, const T *dt,
                       float *g, float *betao, Layout p) {
  const size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= p.total) return;
  const size_t h = linear % p.hidden, s = (linear / p.hidden) % p.seq;
  const size_t batch = linear / p.hidden / p.seq;
  ptrdiff_t offsets[6];
  for (int i = 0; i < 6; ++i)
    offsets[i] = static_cast<ptrdiff_t>(batch) * p.strides[i][0] + static_cast<ptrdiff_t>(s) * p.strides[i][1] + static_cast<ptrdiff_t>(h) * p.strides[i][2];
  const float x = static_cast<float>(a[offsets[1]]) + static_cast<float>(dt[h * p.strides[3][0]]);
  const float bx = p.beta * x;
  const float softplus = bx <= p.threshold ? log1pf(expf(bx)) / p.beta : x;
  g[offsets[4]] = -expf(static_cast<float>(A[h * p.strides[0][0]])) * softplus;
  const float bv = static_cast<float>(b[offsets[2]]);
  const float z = expf(bv >= 0 ? -bv : bv);
  betao[offsets[5]] = bv >= 0 ? 1.0f / (1.0f + z) : z / (1.0f + z);
}
}

template <>
class Operator<FusedGatedDeltaNetGating, Device::Type::kNvidia>
    : public FusedGatedDeltaNetGating {
 public:
  Operator(Tensor A, Tensor a, Tensor b, Tensor dt, float beta, float threshold,
           Tensor g, Tensor betao)
      : FusedGatedDeltaNetGating(A, a, b, dt, beta, threshold, g, betao) {
    if (device_.type() != Device::Type::kNvidia)
      throw std::invalid_argument("gating requires NVIDIA device");
    check_device();
    if (total_ / 256 + (total_ % 256 != 0) > std::numeric_limits<unsigned int>::max())
      throw std::overflow_error("gating grid exceeds CUDA limit");
    p_.total = total_; p_.seq = seq_; p_.hidden = hidden_;
    p_.beta = beta; p_.threshold = threshold;
    for (size_t i = 0; i < 6; ++i)
      for (size_t j = 0; j < strides_[i].size(); ++j) p_.strides[i][j] = strides_[i][j];
  }
  // All metadata is a kernel value argument. There is no allocation, host copy,
  // synchronization or data-dependent host work in operator(); stream launch
  // is capturable. Make owns no device resources and requires zero workspace.
  bool capture_safe() const override { return true; }
  void operator()(Tensor A, Tensor a, Tensor b, Tensor dt, float beta, float threshold,
                  Tensor g, Tensor betao) const override {
    check_device();
    std::array<Tensor, 6> tensors{A, a, b, dt, g, betao};
    for (size_t i = 0; i < 6; ++i)
      if (tensors[i].device() != device_ || tensors[i].shape() != shapes_[i] ||
          tensors[i].strides() != strides_[i] || tensors[i].dtype() != types_[i])
        throw std::invalid_argument("gating metadata changed after Make");
    if (std::memcmp(&beta, &beta_, sizeof(float)) || std::memcmp(&threshold, &threshold_, sizeof(float)))
      throw std::invalid_argument("gating attributes changed after Make");
    if (!total_) return;
    const dim3 grid(static_cast<unsigned int>(total_ / 256 + (total_ % 256 != 0)));
    const auto stream = static_cast<cudaStream_t>(stream_ ? stream_ : nullptr);
    switch (types_[1]) {
      case DataType::kFloat32: launch<float>(A, a, b, dt, g, betao, grid, stream); break;
      case DataType::kFloat16: launch<half>(A, a, b, dt, g, betao, grid, stream); break;
      case DataType::kBFloat16: launch<__nv_bfloat16>(A, a, b, dt, g, betao, grid, stream); break;
      default: throw std::invalid_argument("gating dtype unsupported");
    }
    if (cudaPeekAtLastError() != cudaSuccess) throw std::runtime_error("gating CUDA launch failed");
  }
 private:
  gating_detail::Layout p_{};
  void check_device() const {
    int current = -1;
    if (cudaGetDevice(&current) != cudaSuccess) throw std::runtime_error("gating device query failed");
    if (current != device_.index()) throw std::invalid_argument("gating owner device mismatch");
  }
  template <typename T>
  void launch(Tensor A, Tensor a, Tensor b, Tensor dt, Tensor g, Tensor betao,
              dim3 grid, cudaStream_t stream) const {
    gating_detail::kernel<T><<<grid, 256, 0, stream>>>(
        static_cast<const T *>(A.data()), static_cast<const T *>(a.data()),
        static_cast<const T *>(b.data()), static_cast<const T *>(dt.data()),
        static_cast<float *>(g.data()), static_cast<float *>(betao.data()), p_);
  }
};
}
