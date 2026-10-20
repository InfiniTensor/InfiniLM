#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <climits>

#include "base/causal_conv1d.h"
#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"

namespace infini::ops {
namespace causal_conv1d_detail {
struct Layout {
  ptrdiff_t strides[9][3]{};
  size_t tokens, channels, pool;
  bool cu_i64, initial_i64, final_i64;
};
__device__ inline int64_t index(const void* p, bool i64, int i, int fallback) {
  return p ? (i64 ? static_cast<const int64_t*>(p)[i]
                  : static_cast<int64_t>(static_cast<const int32_t*>(p)[i]))
           : fallback;
}
template <typename T>
__device__ inline float history(const T* state, const T* qkv, int64_t pos,
                                int64_t begin, int batch, int channel,
                                ptrdiff_t base, const Layout& p) {
  if (pos < 3) return static_cast<float>(state[base + pos * p.strides[2][2]]);
  const auto off = static_cast<ptrdiff_t>(batch) * p.strides[0][0] +
                   static_cast<ptrdiff_t>(begin + pos - 3) * p.strides[0][1] +
                   static_cast<ptrdiff_t>(channel) * p.strides[0][2];
  return static_cast<float>(qkv[off]);
}
// Port of baseline InfiniCore-0.2.9 causal_conv1d_k4_kernel. One thread per
// request/channel reads the entire sequence before committing its final state.
template <typename T>
__global__ void kernel(const T* qkv, const T* weight, T* state, const T* bias,
                       const void* cu, const void* initial,
                       const void* final_indices, T* out, T* final_state,
                       Layout p) {
  const int request = blockIdx.x, channel = blockIdx.y;
  if (static_cast<size_t>(channel) >= p.channels || threadIdx.x != 0) return;
  int64_t begin = 0, end = static_cast<int64_t>(p.tokens);
  int batch = request;
  if (cu) {
    begin = index(cu, p.cu_i64, request, 0);
    end = index(cu, p.cu_i64, request + 1, 0);
    batch = 0;
    // Preserve baseline device-value behavior: malformed range leaves the
    // corresponding output/state untouched. This is not host validation.
    if (begin < 0 || end < begin || end > static_cast<int64_t>(p.tokens))
      return;
  }
  const int64_t length = end - begin;
  const int64_t read = index(initial, p.initial_i64, request, request);
  const int64_t write = index(final_indices, p.final_i64, request, request);
  if (read < 0 || write < 0 || read >= static_cast<int64_t>(p.pool) ||
      (final_indices && write >= static_cast<int64_t>(p.pool)))
    return;
  const auto base = static_cast<ptrdiff_t>(read) * p.strides[2][0] +
                    static_cast<ptrdiff_t>(channel) * p.strides[2][1];
  const auto wb = static_cast<ptrdiff_t>(channel) * p.strides[1][0];
  for (int64_t t = 0; t < length; ++t) {
    float acc = static_cast<float>(weight[wb]) *
                    history(state, qkv, t, begin, batch, channel, base, p) +
                static_cast<float>(weight[wb + p.strides[1][2]]) *
                    history(state, qkv, t + 1, begin, batch, channel, base, p) +
                static_cast<float>(weight[wb + 2 * p.strides[1][2]]) *
                    history(state, qkv, t + 2, begin, batch, channel, base, p) +
                static_cast<float>(weight[wb + 3 * p.strides[1][2]]) *
                    history(state, qkv, t + 3, begin, batch, channel, base, p);
    if (bias)
      acc += static_cast<float>(
          bias[static_cast<ptrdiff_t>(channel) * p.strides[3][0]]);
    const auto off = static_cast<ptrdiff_t>(batch) * p.strides[7][0] +
                     static_cast<ptrdiff_t>(begin + t) * p.strides[7][1] +
                     static_cast<ptrdiff_t>(channel) * p.strides[7][2];
    out[off] = static_cast<T>(acc);
  }
  T* target = final_indices ? state : final_state;
  const int stride_index = final_indices ? 2 : 8;
  const auto target_base =
      static_cast<ptrdiff_t>(final_indices ? write : request) *
          p.strides[stride_index][0] +
      static_cast<ptrdiff_t>(channel) * p.strides[stride_index][1];
  // Same ordered commit as baseline. Exact initial/final alias is valid for
  // short sequences because copied history source positions lie ahead of
  // writes.
  target[target_base] = static_cast<T>(
      history(state, qkv, length, begin, batch, channel, base, p));
  target[target_base + p.strides[stride_index][2]] = static_cast<T>(
      history(state, qkv, length + 1, begin, batch, channel, base, p));
  target[target_base + 2 * p.strides[stride_index][2]] = static_cast<T>(
      history(state, qkv, length + 2, begin, batch, channel, base, p));
}
}  // namespace causal_conv1d_detail

template <>
class Operator<CausalConv1d, Device::Type::kNvidia> : public CausalConv1d {
 public:
  Operator(const Tensor qkv, const Tensor weight, Tensor conv_state,
           const std::optional<Tensor> bias,
           const std::optional<Tensor> cu_seqlens,
           const std::optional<Tensor> initial_state_indices,
           const std::optional<Tensor> final_state_indices, Tensor out,
           std::optional<Tensor> final_conv_state)
      : CausalConv1d(qkv, weight, conv_state, bias, cu_seqlens,
                     initial_state_indices, final_state_indices, out,
                     final_conv_state) {
    if (device_.type() != Device::Type::kNvidia)
      throw std::invalid_argument("causal conv requires NVIDIA");
    if (weight.shape()[2] != 4 || conv_state.shape()[2] != 3)
      throw std::invalid_argument(
          "NVIDIA causal conv kernel requires K4 and state_len3");
    if (tokens_ > static_cast<size_t>(std::numeric_limits<int64_t>::max() - 3))
      throw std::overflow_error(
          "NVIDIA causal conv history index exceeds int64");
    check_device();
    // Baseline uses int request/channel variables and CUDA's grid.y dimension.
    if (requests_ >= static_cast<size_t>(INT_MAX) || channels_ > 65535)
      throw std::invalid_argument(
          "causal conv request/channel launch extent unsupported");
    p_.tokens = tokens_;
    p_.channels = channels_;
    p_.pool = pool_;
    for (size_t i = 0; i < metadata_.size(); ++i)
      if (metadata_[i])
        for (size_t j = 0; j < metadata_[i]->strides().size(); ++j)
          p_.strides[i][j] = metadata_[i]->strides()[j];
    p_.cu_i64 = metadata_[4] && metadata_[4]->dtype() == DataType::kInt64;
    p_.initial_i64 = metadata_[5] && metadata_[5]->dtype() == DataType::kInt64;
    p_.final_i64 = metadata_[6] && metadata_[6]->dtype() == DataType::kInt64;
  }
  // Device capture/state-pool lifetime is not supported for this stateful path.
  // Zero device allocations does not alone establish capture eligibility.
  void operator()(const Tensor qkv, const Tensor weight, Tensor conv_state,
                  const std::optional<Tensor> bias,
                  const std::optional<Tensor> cu_seqlens,
                  const std::optional<Tensor> initial_state_indices,
                  const std::optional<Tensor> final_state_indices, Tensor out,
                  std::optional<Tensor> final_conv_state) const override {
    const Views views{qkv,
                      weight,
                      conv_state,
                      bias,
                      cu_seqlens,
                      initial_state_indices,
                      final_state_indices,
                      out,
                      final_conv_state};
    for (size_t i = 0; i < views.size(); ++i) {
      const auto& t = views[i];
      if (t.has_value() != metadata_[i].has_value() ||
          (t && (!std::equal_to<Tensor>{}(*t, *metadata_[i]) ||
                 (t->numel() && !t->data()))))
        throw std::invalid_argument(
            "causal conv metadata changed or nonempty storage is null");
    }
    check_device();
    if (!requests_ || !channels_) return;
    switch (dtype_) {
      case DataType::kFloat32:
        launch<float>(views);
        break;
      case DataType::kFloat16:
        launch<half>(views);
        break;
      case DataType::kBFloat16:
        launch<__nv_bfloat16>(views);
        break;
      default:
        throw std::invalid_argument("causal conv dtype unsupported");
    }
    if (cudaPeekAtLastError() != cudaSuccess)
      throw std::runtime_error("causal conv CUDA launch failed");
  }

 private:
  causal_conv1d_detail::Layout p_{};
  void check_device() const {
    int current = -1;
    if (cudaGetDevice(&current) != cudaSuccess)
      throw std::runtime_error("causal conv device query failed");
    if (current != device_.index())
      throw std::invalid_argument("causal conv owner device mismatch");
  }
  template <typename T>
  void launch(const Views& v) const {
    auto pointer = [&](size_t i) -> void* {
      return v[i] ? const_cast<void*>(v[i]->data()) : nullptr;
    };
    const dim3 grid(static_cast<unsigned int>(requests_),
                    static_cast<unsigned int>(channels_), 1);
    const auto stream = static_cast<cudaStream_t>(stream_ ? stream_ : nullptr);
    causal_conv1d_detail::kernel<T><<<grid, 1, 0, stream>>>(
        static_cast<const T*>(pointer(0)), static_cast<const T*>(pointer(1)),
        static_cast<T*>(pointer(2)), static_cast<const T*>(pointer(3)),
        pointer(4), pointer(5), pointer(6), static_cast<T*>(pointer(7)),
        static_cast<T*>(pointer(8)), p_);
  }
};
}  // namespace infini::ops
