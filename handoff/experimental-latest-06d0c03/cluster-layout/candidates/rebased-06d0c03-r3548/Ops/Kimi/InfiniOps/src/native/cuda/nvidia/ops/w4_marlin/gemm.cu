#include <cuda_bf16.h>

#include "kernel.h"
#include "launch.h"
#include "core/scalar_type.hpp"
#include "marlin.cuh"

namespace infini::ops::native_w4 {
inline std::size_t align256(std::size_t bytes) {
  if (bytes > std::numeric_limits<std::size_t>::max()-255)
    throw std::overflow_error("W4 workspace alignment overflow");
  return (bytes+255)&~std::size_t(255);
}
inline std::size_t append(std::size_t& total, std::size_t size) {
  const auto result = align256(total);
  if (size > std::numeric_limits<std::size_t>::max()-result)
    throw std::overflow_error("W4 workspace sum overflow");
  total = result + size;
  return result;
}
template <typename T>
__global__ void epilogue(T* output, const T* bias, std::size_t count, int n, float alpha) {
  const std::size_t i = std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if (i < count) {
    const float value = float(output[i])*alpha + (bias ? float(bias[i%n]) : 0.f);
    output[i] = T(value);
  }
}
} // namespace infini::ops::native_w4

namespace infini::ops {
Operator<W4MarlinGemm, Device::Type::kNvidia, 0>::Operator(
    const Tensor a, const Tensor b, const Tensor scales, const Tensor zeros,
    const Tensor g_idx, const Tensor perm, const Tensor bias,
    const int64_t scheme, const int64_t group_size, const bool is_k_full,
    const float alpha, Tensor out)
    : W4MarlinGemm(a,b,scales,zeros,g_idx,perm,bias,scheme,group_size,is_k_full,alpha,out), device_{a.device().index()} {
  w4::require(a.device().type() == Device::Type::kNvidia, "W4 requires one NVIDIA device");
  if (!is_k_full && perm.numel() && g_idx.data()) w4::address(g_idx, 16);
  int shared; native_w4::info(device_,sms_,shared);
  // Exactly the original FP32 reduction requirement plus A permutation temp.
  const auto max_m = std::min<int64_t>(((m_+15)/16)*16,64);
  ctmp_ = native_w4::append(bytes_,w4::mul(w4::mul(sms_,max_m),marlin::max_thread_n*sizeof(float)));
  atmp_ = native_w4::append(bytes_,perm.numel() ? w4::mul(w4::mul(m_,k_),a.element_size()) : 0);
  locks_ = native_w4::append(bytes_,w4::mul(sms_,sizeof(int)));
  bytes_ = native_w4::align256(bytes_);
}
void Operator<W4MarlinGemm, Device::Type::kNvidia, 0>::operator()(
    const Tensor a, const Tensor b, const Tensor scales, const Tensor zeros,
    const Tensor g_idx, const Tensor perm, const Tensor bias,
    const int64_t scheme, const int64_t group_size, const bool is_k_full,
    const float alpha, Tensor out) const {
  w4::require(scheme == scheme_ && group_size == group_ && is_k_full == full_ && alpha == alpha_, "W4 GEMM attributes changed");
  for (const auto& t : {a,b,scales,zeros,out}) w4::address(t,16);
  for (const auto& t : {g_idx,perm,bias}) w4::address(t,4);
  if (!full_ && perm.numel()) w4::address(g_idx,16);
  for (const auto& t : {a,b,scales,zeros,g_idx,perm,bias}) w4::disjoint(out,t);
  w4::require(workspace_ && workspace_size_in_bytes_ >= bytes_ &&
              reinterpret_cast<std::uintptr_t>(workspace_)%256 == 0, "W4 GEMM requires aligned external workspace");
  for (const auto& t : {a,b,scales,zeros,g_idx,perm,bias,out})
    w4::disjoint(workspace_,bytes_,t.data(),w4::span(t));
  native_w4::DeviceGuard guard(device_);
  auto stream = static_cast<cudaStream_t>(stream_);
  native_w4::no_capture(stream);
  auto* memory = static_cast<std::byte*>(workspace_);
  // Locks are state, not merely scratch. Every eager/replayed call resets them
  // on the same stream before the original reduction kernel consumes them.
  native_w4::check(cudaMemsetAsync(memory+locks_,0,sms_*sizeof(int),stream), "W4 lock reset");
  const auto type = a.dtype() == DataType::kFloat16 ? vllm::kFloat16 : vllm::kBFloat16;
  const auto btype = scheme == 0 ? vllm::kU4 : vllm::kU4B8;
  const bool act = perm.numel() != 0;
  const int group = act && !full_ ? 0 : (scales.size(0) == 1 ? -1 : int(group_));
  // The original unified AWQ kernel generator already contains GPTQ U4B8
  // specializations. This avoids the old separate GPTQ TVM-FFI tensor wrapper.
  marlin::marlin_mm(a.data(),b.data(),out.data(),memory+ctmp_,nullptr,
      nullptr,const_cast<void*>(scales.data()),nullptr,
      zeros.numel() ? const_cast<void*>(zeros.data()) : nullptr,
      act ? const_cast<void*>(g_idx.data()) : nullptr,
      act ? const_cast<void*>(perm.data()) : nullptr,
      act ? memory+atmp_ : nullptr,m_,n_,k_,a.stride(0),memory+locks_,
      type,btype,type,type,false,act,full_,scheme == 0,scales.size(0),group,
      device_,stream,-1,-1,sms_,false,true,false);
  native_w4::check(cudaGetLastError(), "W4 Marlin GEMM launch");
  if (alpha != 1.f || bias.numel()) {
    const auto count = out.numel();
    if (a.dtype() == DataType::kFloat16)
      native_w4::epilogue<<<(count+255)/256,256,0,stream>>>(static_cast<half*>(out.data()),bias.numel() ? static_cast<const half*>(bias.data()) : nullptr,count,n_,alpha);
    else
      native_w4::epilogue<<<(count+255)/256,256,0,stream>>>(static_cast<nv_bfloat16*>(out.data()),bias.numel() ? static_cast<const nv_bfloat16*>(bias.data()) : nullptr,count,n_,alpha);
    native_w4::check(cudaGetLastError(), "W4 alpha/bias epilogue");
  }
}
} // namespace infini::ops
