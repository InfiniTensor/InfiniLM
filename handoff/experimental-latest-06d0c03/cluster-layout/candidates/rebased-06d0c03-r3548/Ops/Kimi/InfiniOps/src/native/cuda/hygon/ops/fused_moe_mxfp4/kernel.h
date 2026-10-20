#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "base/fused_moe_mxfp4.h"
#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/fused_moe_mxfp4/kernel.cuh"
#include "kimi_mxfp4_dispatch_r330.cuh"

namespace infini::ops {
template <>
class Operator<FusedMoeMxfp4, Device::Type::kHygon> : public FusedMoeMxfp4 {
 public:
  using FusedMoeMxfp4::FusedMoeMxfp4;
  void operator()(Tensor output, const Tensor input, const Tensor indices,
                  const Tensor routing_weights, const Tensor w13, const Tensor s13,
                  const Tensor w2, const Tensor s2, size_t activation) const override {
    Invocation(output,input,indices,routing_weights,w13,s13,w2,s2,activation);
    if(!geometry_.num_tokens) return;
    moe_detail::Workspace(workspace_,workspace_size_in_bytes_,workspace_bytes_,{output,input,indices,routing_weights,w13,s13,w2,s2});
    switch(dtype_) {
      case DataType::kFloat16: Launch<half>(output,input,indices,routing_weights,w13,s13,w2,s2);break;
      case DataType::kBFloat16: Launch<__nv_bfloat16>(output,input,indices,routing_weights,w13,s13,w2,s2);break;
      case DataType::kFloat32: Launch<float>(output,input,indices,routing_weights,w13,s13,w2,s2);break;
      default: throw std::invalid_argument("Unsupported MXFP4 activation type");
    }
  }
 protected:
  virtual bool ParallelRoutes() const { return false; }
 private:
  template<class T> void Launch(Tensor output, const Tensor input, const Tensor indices,
                               const Tensor routing_weights, const Tensor w13, const Tensor s13,
                               const Tensor w2, const Tensor s2) const {
    mxfp4_detail::launchFusedMoeMxfp4R330(
      reinterpret_cast<T*>(output.data()),reinterpret_cast<T*>(workspace_),
      reinterpret_cast<const T*>(input.data()),reinterpret_cast<const int32_t*>(indices.data()),
      reinterpret_cast<const float*>(routing_weights.data()),reinterpret_cast<const uint8_t*>(w13.data()),
      reinterpret_cast<const uint8_t*>(s13.data()),reinterpret_cast<const uint8_t*>(w2.data()),
      reinterpret_cast<const uint8_t*>(s2.data()),geometry_,reinterpret_cast<cudaStream_t>(stream_),ParallelRoutes());
    const auto status=cudaGetLastError();
    if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
  }
};
template <>
class Operator<FusedMoeMxfp4, Device::Type::kHygon, 1>
    : public Operator<FusedMoeMxfp4, Device::Type::kHygon, 0> {
 public:
  using Parent=Operator<FusedMoeMxfp4,Device::Type::kHygon,0>;
  using Parent::Parent;
 protected:
  bool ParallelRoutes() const override { return true; }
};
} // namespace infini::ops
