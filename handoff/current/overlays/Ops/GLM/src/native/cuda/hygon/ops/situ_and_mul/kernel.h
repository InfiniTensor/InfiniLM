#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "base/situ_and_mul.h"
#include "native/cuda/hygon/runtime_.h"

namespace infini::ops::situ_detail {
struct Geometry {
  size_t count,rank,shape[8],gate_stride[8],up_stride[8];
};
template<class T> __global__ void Kernel(T *out,const T *gate,const T *up,
    Geometry geometry,float beta,float linear_beta) {
  for(size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;index<geometry.count;
      index+=size_t(gridDim.x)*blockDim.x) {
    size_t rem=index,go=0,uo=0;
    for(size_t d=geometry.rank;d-->0;) {
      size_t coordinate=rem%geometry.shape[d];rem/=geometry.shape[d];
      go+=coordinate*geometry.gate_stride[d];uo+=coordinate*geometry.up_stride[d];
    }
    const float g=float(gate[go]),u=float(up[uo]);
    const float eg=expf(g>=0?-g:g);
    const float sigmoid=g>=0?1.f/(1.f+eg):eg/(1.f+eg);
    out[index]=T(beta*tanhf(g/beta)*sigmoid*linear_beta*tanhf(u/linear_beta));
  }
}
} // namespace infini::ops::situ_detail

namespace infini::ops {
template<> class Operator<SituAndMul,Device::Type::kHygon> : public SituAndMul {
 public:
  using SituAndMul::SituAndMul;
  static bool supports(Tensor output,const Tensor gate,const Tensor up,float beta,float linear_beta) {
    try {Validate(output,gate,up,beta,linear_beta);return output.device().type()==Device::Type::kHygon;}
    catch(const std::invalid_argument&) {return false;}
  }
  bool capture_safe() const override {return false;}
  void operator()(Tensor output,const Tensor gate,const Tensor up,float beta,float linear_beta) const override {
    Invocation(output,gate,up,beta,linear_beta);
    if(!output.numel())return;
    situ_detail::Geometry geometry{};geometry.count=output.numel();geometry.rank=output.ndim();
    for(size_t d=0;d<geometry.rank;++d) {
      geometry.shape[d]=output.size(d);geometry.gate_stride[d]=gate.stride(d);geometry.up_stride[d]=up.stride(d);
    }
    switch(output.dtype()) {
      case DataType::kFloat32:Launch<float>(output,gate,up,geometry,beta,linear_beta);break;
      case DataType::kFloat16:Launch<half>(output,gate,up,geometry,beta,linear_beta);break;
      case DataType::kBFloat16:Launch<__nv_bfloat16>(output,gate,up,geometry,beta,linear_beta);break;
      default:throw std::invalid_argument("SiTu dtype unsupported");
    }
    const auto status=cudaGetLastError();if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
  }
 private:
  template<class T> void Launch(Tensor output,const Tensor gate,const Tensor up,
      situ_detail::Geometry geometry,float beta,float linear_beta) const {
    const unsigned blocks=unsigned(std::min<size_t>((geometry.count-1)/256+1,65535));
    situ_detail::Kernel<T><<<blocks,256,0,reinterpret_cast<cudaStream_t>(stream_)>>>(
      reinterpret_cast<T*>(output.data()),reinterpret_cast<const T*>(gate.data()),
      reinterpret_cast<const T*>(up.data()),geometry,beta,linear_beta);
  }
};
} // namespace infini::ops
