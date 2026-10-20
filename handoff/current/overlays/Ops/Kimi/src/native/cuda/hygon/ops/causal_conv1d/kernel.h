#pragma once
#include "base/causal_conv1d.h"
#include "native/cuda/hygon/caster.cuh"
#include "native/cuda/hygon/runtime_.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <climits>
#include "base/moe_compute_common.h"
#include <set>
#include <vector>

namespace infini::ops {
namespace causal_conv1d_detail {
struct Layout {
  ptrdiff_t strides[9][3]{};
  size_t tokens, channels, pool;
  bool cu_i64, initial_i64, final_i64;
};
__device__ inline int64_t index(const void *p, bool i64, int i, int fallback) {
  return p ? (i64 ? static_cast<const int64_t *>(p)[i]
                  : static_cast<int64_t>(static_cast<const int32_t *>(p)[i])) : fallback;
}
template <typename T>
__device__ inline float history(const T *state, const T *qkv, int64_t pos,
    int64_t begin, int batch, int channel, ptrdiff_t base, const Layout &p) {
  if (pos < 3) return static_cast<float>(state[base + pos * p.strides[2][2]]);
  const auto off = static_cast<ptrdiff_t>(batch) * p.strides[0][0] +
      static_cast<ptrdiff_t>(begin + pos - 3) * p.strides[0][1] +
      static_cast<ptrdiff_t>(channel) * p.strides[0][2];
  return static_cast<float>(qkv[off]);
}
// Port of baseline InfiniCore-0.2.9 causal_conv1d_k4_kernel. Channels are coalesced in groups of 128.
// Reads use the immutable caller-workspace state snapshot, including slot swaps.
template <typename T, typename W>
__global__ void kernel(const T *qkv, const W *weight, const T *state, T *mutable_state, const T *bias,
    const void *cu, const void *initial, const void *final_indices,
    T *out, T *final_state, Layout p) {
  const int request = blockIdx.x, channel = blockIdx.y * blockDim.x + threadIdx.x;
  if (static_cast<size_t>(channel) >= p.channels) return;
  int64_t begin = 0, end = static_cast<int64_t>(p.tokens);
  int batch = request;
  if (cu) {
    begin = index(cu, p.cu_i64, request, 0);
    end = index(cu, p.cu_i64, request + 1, 0);
    batch = 0;
    // Preserve baseline device-value behavior: malformed range leaves the
    // corresponding output/state untouched. This is not host validation.
    if (begin < 0 || end < begin || end > static_cast<int64_t>(p.tokens)) return;
  }
  const int64_t length = end - begin;
  const int64_t read = index(initial, p.initial_i64, request, request);
  const int64_t write = index(final_indices, p.final_i64, request, request);
  if (read < 0 || write < 0 || read >= static_cast<int64_t>(p.pool) ||
      (final_indices && write >= static_cast<int64_t>(p.pool))) return;
  const auto base = static_cast<ptrdiff_t>(read) * p.strides[2][0] +
      static_cast<ptrdiff_t>(channel) * p.strides[2][1];
  const auto wb = static_cast<ptrdiff_t>(channel) * p.strides[1][0];
  for (int64_t t = 0; t < length; ++t) {
    float acc = static_cast<float>(weight[wb]) * history(state, qkv, t, begin, batch, channel, base, p)
        + static_cast<float>(weight[wb + p.strides[1][2]]) * history(state, qkv, t + 1, begin, batch, channel, base, p)
        + static_cast<float>(weight[wb + 2 * p.strides[1][2]]) * history(state, qkv, t + 2, begin, batch, channel, base, p)
        + static_cast<float>(weight[wb + 3 * p.strides[1][2]]) * history(state, qkv, t + 3, begin, batch, channel, base, p);
    if (bias) acc += static_cast<float>(bias[static_cast<ptrdiff_t>(channel) * p.strides[3][0]]);
    const auto off = static_cast<ptrdiff_t>(batch) * p.strides[7][0] +
        static_cast<ptrdiff_t>(begin + t) * p.strides[7][1] +
        static_cast<ptrdiff_t>(channel) * p.strides[7][2];
    out[off] = static_cast<T>(acc);
  }
  T *target = final_indices ? mutable_state : final_state;
  const int stride_index = final_indices ? 2 : 8;
  const auto target_base = static_cast<ptrdiff_t>(final_indices ? write : request) * p.strides[stride_index][0] +
      static_cast<ptrdiff_t>(channel) * p.strides[stride_index][1];
  // Same ordered commit as baseline. Exact initial/final alias is valid for
  // short sequences because copied history source positions lie ahead of writes.
  target[target_base] = static_cast<T>(history(state, qkv, length, begin, batch, channel, base, p));
  target[target_base + p.strides[stride_index][2]] = static_cast<T>(history(state, qkv, length + 1, begin, batch, channel, base, p));
  target[target_base + 2 * p.strides[stride_index][2]] = static_cast<T>(history(state, qkv, length + 2, begin, batch, channel, base, p));
}
}

template <>
class Operator<CausalConv1d, Device::Type::kHygon> : public CausalConv1d {
 public:
  Operator(const Tensor qkv, const Tensor weight, Tensor conv_state,
      const std::optional<Tensor> bias, const std::optional<Tensor> cu_seqlens,
      const std::optional<Tensor> initial_state_indices,
      const std::optional<Tensor> final_state_indices, Tensor out,
      std::optional<Tensor> final_conv_state)
      : CausalConv1d(qkv, weight, conv_state, bias, cu_seqlens,
                    initial_state_indices, final_state_indices, out, final_conv_state) {
    if (device_.type() != Device::Type::kHygon) throw std::invalid_argument("causal conv requires HYGON");
    check_device();
    Verify(Views{qkv, weight, conv_state, bias, cu_seqlens, initial_state_indices, final_state_indices, out, final_conv_state});
    workspace_bytes_ = moe_detail::Product(conv_state.numel(),conv_state.element_size());
    // Baseline uses int request/channel variables and CUDA's grid.y dimension.
    if (requests_ > 65535 || channels_ > 65535 * 128 || tokens_ > size_t(INT_MAX-3))
      throw std::invalid_argument("causal conv request/channel launch extent unsupported");
    p_.tokens = tokens_; p_.channels = channels_; p_.pool = pool_;
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
  bool capture_safe() const override { return false; }
  size_t workspace_size_in_bytes() const override { return workspace_bytes_; }
  void operator()(const Tensor qkv, const Tensor weight, Tensor conv_state,
      const std::optional<Tensor> bias, const std::optional<Tensor> cu_seqlens,
      const std::optional<Tensor> initial_state_indices,
      const std::optional<Tensor> final_state_indices, Tensor out,
      std::optional<Tensor> final_conv_state) const override {
    const Views views{qkv, weight, conv_state, bias, cu_seqlens,
                      initial_state_indices, final_state_indices, out, final_conv_state};
    Validate(views); Verify(views); Same(views); check_device();
    if (!requests_ || !channels_) return;
    Prepare(views);
    switch (dtype_) {
      case DataType::kFloat32: if(weight.dtype()==DataType::kFloat32) launch<float,float>(views); else launch<float,float>(views); break;
      case DataType::kFloat16: if(weight.dtype()==DataType::kFloat32) launch<half,float>(views); else launch<half,half>(views); break;
      case DataType::kBFloat16: if(weight.dtype()==DataType::kFloat32) launch<__nv_bfloat16,float>(views); else launch<__nv_bfloat16,__nv_bfloat16>(views); break;
      default: throw std::invalid_argument("causal conv dtype unsupported");
    }
    if (cudaPeekAtLastError() != cudaSuccess) throw std::runtime_error("causal conv CUDA launch failed");
  }
 private:
  causal_conv1d_detail::Layout p_{};
  size_t workspace_bytes_{};
  static size_t Span(const Tensor &t) {
    size_t span=0;
    for(size_t i=0;i<t.ndim();++i){
      if(t.stride(i)<0)throw std::invalid_argument("causal conv negative strides unsupported");
      if(!t.size(i))return 0;
      auto term=moe_detail::Product(t.size(i)-1,size_t(t.stride(i)));
      if(term>SIZE_MAX-span)throw std::overflow_error("causal conv span overflow");
      span+=term;
    }
    if(span==SIZE_MAX)throw std::overflow_error("causal conv span overflow");
    return moe_detail::Product(span+1,t.element_size());
  }
  static bool Overlap(const Tensor &a,const Tensor &b){
    auto an=Span(a),bn=Span(b),ap=reinterpret_cast<uintptr_t>(a.data()),bp=reinterpret_cast<uintptr_t>(b.data());
    return an&&bn&&(ap<=bp?bp-ap<an:ap-bp<bn);
  }
  static void Verify(const Views &v){
    if(v[0]->device().type()!=Device::Type::kHygon)throw std::invalid_argument("causal conv requires HYGON");
    for(size_t i=0;i<v.size();++i)if(v[i]){
      auto span=Span(*v[i]);
      if(span&&(!v[i]->data()||reinterpret_cast<uintptr_t>(v[i]->data())%v[i]->element_size()))throw std::invalid_argument("causal conv invalid pointer alignment");
      // Non-overlapping contiguous destinations establish safe state snapshot/commit.
      if(span&&(i==2||i==7||i==8)&&!v[i]->IsContiguous())throw std::invalid_argument("causal conv destinations must be contiguous");
      for(size_t j=0;j<i;++j)if(v[j]&&(i==2||i==7||i==8||j==2||j==7||j==8)&&Overlap(*v[i],*v[j]))throw std::invalid_argument("causal conv destination aliases operands");
    }
  }
  void Same(const Views &v) const {
    for(size_t i=0;i<v.size();++i){
      if(v[i].has_value()!=metadata_[i].has_value())throw std::invalid_argument("causal conv cached optional metadata mismatch");
      if(v[i]&&(v[i]->shape()!=metadata_[i]->shape()||v[i]->strides()!=metadata_[i]->strides()||v[i]->dtype()!=metadata_[i]->dtype()||v[i]->device()!=device_))throw std::invalid_argument("causal conv cached metadata mismatch");
    }
  }
  static void Check(cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));}
  std::vector<int64_t> Indices(const std::optional<Tensor> &tensor,size_t count) const {
    std::vector<int64_t> result(count);
    if(!tensor){for(size_t i=0;i<count;++i)result[i]=int64_t(i);return result;}
    auto stream=static_cast<cudaStream_t>(stream_);
    if(tensor->dtype()==DataType::kInt64){Check(cudaMemcpyAsync(result.data(),tensor->data(),count*sizeof(int64_t),cudaMemcpyDeviceToHost,stream));Check(cudaStreamSynchronize(stream));}
    else {std::vector<int32_t> small(count);Check(cudaMemcpyAsync(small.data(),tensor->data(),count*sizeof(int32_t),cudaMemcpyDeviceToHost,stream));Check(cudaStreamSynchronize(stream));std::copy(small.begin(),small.end(),result.begin());}
    return result;
  }
  void Prepare(const Views &v) const {
    if(!workspace_||workspace_size_in_bytes_<workspace_bytes_||reinterpret_cast<uintptr_t>(workspace_)%256)throw std::invalid_argument("causal conv caller workspace missing or unaligned");
    auto ap=reinterpret_cast<uintptr_t>(workspace_);
    for(const auto &t:v)if(t){auto span=Span(*t),bp=reinterpret_cast<uintptr_t>(t->data());if(span&&(ap<=bp?bp-ap<workspace_bytes_:ap-bp<span))throw std::invalid_argument("causal conv workspace aliases operand");}
    if(v[4]){auto offsets=Indices(v[4],requests_+1);
      if(offsets.front()!=0||offsets.back()!=int64_t(tokens_))throw std::invalid_argument("causal conv offsets must cover packed tokens");
      for(size_t i=0;i<requests_;++i)if(offsets[i]<0||offsets[i]>offsets[i+1])throw std::invalid_argument("causal conv offsets not monotonic");}
    auto reads=Indices(v[5],requests_),writes=Indices(v[6],requests_);std::set<int64_t> unique;
    for(size_t i=0;i<requests_;++i){
      if(reads[i]<0||reads[i]>=int64_t(pool_))throw std::invalid_argument("causal conv read slot out of range");
      if(v[6]&&(writes[i]<0||writes[i]>=int64_t(pool_)||!unique.insert(writes[i]).second))throw std::invalid_argument("causal conv writes require distinct valid slots");
    }
    Check(cudaMemcpyAsync(workspace_,v[2]->data(),workspace_bytes_,cudaMemcpyDeviceToDevice,static_cast<cudaStream_t>(stream_)));
  }
  void check_device() const {
    int current = -1;
    if (cudaGetDevice(&current) != cudaSuccess) throw std::runtime_error("causal conv device query failed");
    if (current != device_.index()) throw std::invalid_argument("causal conv owner device mismatch");
  }
  template <typename T,typename W>
  void launch(const Views &v) const {
    auto pointer = [&](size_t i) -> void * { return v[i] ? const_cast<void *>(v[i]->data()) : nullptr; };
    const dim3 grid(static_cast<unsigned int>(requests_), static_cast<unsigned int>((channels_+127)/128), 1);
    const auto stream = static_cast<cudaStream_t>(stream_ ? stream_ : nullptr);
    causal_conv1d_detail::kernel<T,W><<<grid, 128, 0, stream>>>(
        static_cast<const T *>(pointer(0)), static_cast<const W *>(pointer(1)),
        static_cast<const T *>(workspace_), static_cast<T *>(pointer(2)), static_cast<const T *>(pointer(3)),
        pointer(4), pointer(5), pointer(6), static_cast<T *>(pointer(7)), static_cast<T *>(pointer(8)), p_);
  }
};
}
