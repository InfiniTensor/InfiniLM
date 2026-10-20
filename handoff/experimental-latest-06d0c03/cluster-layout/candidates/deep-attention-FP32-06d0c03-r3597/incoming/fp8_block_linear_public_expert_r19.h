#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "base/fp8_block_linear.h"
#include "native/cuda/hygon/runtime_.h"

namespace infini::ops::fp8_block_detail {
__device__ inline float Decode(uint8_t value) {
  const unsigned exp=(value>>3)&15, mant=value&7;
  float result=exp ? (exp==15 && mant==7 ? nanf("") : ldexpf(1.0f+float(mant)/8,int(exp)-7)) : ldexpf(float(mant),-9);
  return value&128 ? -result : result;
}
template<class T> __global__ void Linear(T* output,const uint8_t* x,const float* xs,
    const uint8_t* weight,const float* ws,size_t m,size_t n,size_t k,size_t scale_columns,
    size_t row_offset,size_t column_offset) {
  const size_t row=blockIdx.x, token=blockIdx.y;
  if(row>=n || token>=m)return;
  __shared__ float partial[256];float result=0.0f;
  const size_t begin=column_offset/128, end=(column_offset+k+127)/128;
  for(size_t block=begin;block<end;++block) {
    const size_t global_column=block*128+threadIdx.x;
    float value=0.0f;
    if(threadIdx.x<128 && global_column>=column_offset && global_column-column_offset<k) {
      const size_t local_column=global_column-column_offset;
      value=Decode(x[token*k+local_column])*Decode(weight[row*k+local_column]);
    }
    partial[threadIdx.x]=value;__syncthreads();
    for(unsigned step=128;step;step>>=1) {
      if(threadIdx.x<step)partial[threadIdx.x]+=partial[threadIdx.x+step];
      __syncthreads();
    }
    if(threadIdx.x==0)result+=partial[0]*xs[token*scale_columns+block]*ws[((row_offset+row)/128)*scale_columns+block];
    __syncthreads();
  }
  if(threadIdx.x==0)output[token*n+row]=static_cast<T>(result);
}
} // namespace infini::ops::fp8_block_detail

// Private experiment: reproduce the original active128 reduction tree in
// 32/64 physical lanes. FP8 Decode and block scale/accumulation are unchanged.
namespace infini::ops::fp8_block_detail {
template<class T,int Width>
__global__ void LinearWarpR17(T*out,const uint8_t*x,const float*xs,
 const uint8_t*w,const float*ws,size_t m,size_t n,size_t k,size_t scale_columns,
 size_t row_offset,size_t column_offset){
 constexpr int Virtual=128/Width,Groups=256/Width;
 size_t row=size_t(blockIdx.x)*Groups+threadIdx.x/Width,token=blockIdx.y;
 if(row>=n||token>=m)return;
 unsigned lane=threadIdx.x%Width;float result=0.f;
 for(size_t block=column_offset/128;block<(column_offset+k+127)/128;++block){
  float values[Virtual]{};
  #pragma unroll
  for(int j=0;j<Virtual;++j){
   size_t global_column=block*128+lane+size_t(j)*Width;
   if(global_column>=column_offset&&global_column-column_offset<k){
    size_t column=global_column-column_offset;
    values[j]=Decode(x[token*k+column])*Decode(w[row*k+column]);
   }
   // Original stride128 combines each active lane with an explicit zero.
   values[j]=__fadd_rn(values[j],0.f);
  }
  #pragma unroll
  for(int stride=Virtual/2;stride;stride/=2){
   #pragma unroll
   for(int j=0;j<Virtual;++j)if(j<stride)values[j]=__fadd_rn(values[j],values[j+stride]);
  }
  float value=values[0];
  #pragma unroll
  for(int shift=Width/2;shift;shift/=2){
#if defined(__HIPCC__)
   value=__fadd_rn(value,__shfl_down(value,shift,Width));
#else
   value=__fadd_rn(value,__shfl_down_sync(0xffffffffu,value,shift,Width));
#endif
  }
  if(lane==0)result+=value*xs[token*scale_columns+block]*ws[((row_offset+row)/128)*scale_columns+block];
 }
 if(lane==0)out[token*n+row]=static_cast<T>(result);
}
}


namespace infini::ops {
template<> class Operator<Fp8BlockLinear,Device::Type::kHygon> : public Fp8BlockLinear {
 public:
  using Fp8BlockLinear::Fp8BlockLinear;
  void operator()(Tensor output,const Tensor x,const Tensor x_scale,const Tensor weight,
                  const Tensor weight_scale,size_t row_offset,size_t column_offset) const override {
    Invocation(output,x,x_scale,weight,weight_scale,row_offset,column_offset);
    if(!x.size(0))return;
    switch(output.dtype()) {
      case DataType::kFloat32:Launch<float>(output,x,x_scale,weight,weight_scale);break;
      case DataType::kFloat16:Launch<half>(output,x,x_scale,weight,weight_scale);break;
      case DataType::kBFloat16:Launch<__nv_bfloat16>(output,x,x_scale,weight,weight_scale);break;
      default:throw std::invalid_argument("FP8 block linear output dtype unsupported");
    }
    const auto status=cudaGetLastError();if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
  }
 private:
  template<class T> void Launch(Tensor output,const Tensor x,const Tensor x_scale,const Tensor weight,const Tensor weight_scale) const {
    // gfx936 candidate: retain the scalar launch for small grids. The
    // original FP8 Decode, global scales, reduction tree and output cast stay.
    const auto m=x.size(0),n=weight.size(0),k=x.size(1);
    // Actual TP8 down projections are wide-output, short-K grids. Width32
    // avoids the scalar CTA barriers even at one token. The narrow expert
    // gate/up rows benefit only from m>=4; preserve their m1/m2 fallback.
    if(n>=4096 && k>=256 && k<=1536){
      Warp<T,32>(output,x,x_scale,weight,weight_scale);return;
    }
    if(n==256 && k>=768 && m>=4){
      Warp<T,64>(output,x,x_scale,weight,weight_scale);return;
    }
    if(n>=512 && k>=768 && m*n>=1024){
      if(m>=16)Warp<T,32>(output,x,x_scale,weight,weight_scale);
      else Warp<T,64>(output,x,x_scale,weight,weight_scale);
      return;
    }
    fp8_block_detail::Linear<T><<<dim3(weight.size(0),x.size(0)),256,0,reinterpret_cast<cudaStream_t>(stream_)>>>(
      reinterpret_cast<T*>(output.data()),reinterpret_cast<const uint8_t*>(x.data()),reinterpret_cast<const float*>(x_scale.data()),
      reinterpret_cast<const uint8_t*>(weight.data()),reinterpret_cast<const float*>(weight_scale.data()),
      x.size(0),weight.size(0),x.size(1),weight_scale.size(1),row_offset_,column_offset_);
  }
  template<class T,int Width>void Warp(Tensor output,const Tensor x,const Tensor xs,const Tensor w,const Tensor ws)const{
    fp8_block_detail::LinearWarpR17<T,Width><<<dim3(unsigned((w.size(0)+(256/Width)-1)/(256/Width)),unsigned(x.size(0))),256,0,reinterpret_cast<cudaStream_t>(stream_)>>>(reinterpret_cast<T*>(output.data()),reinterpret_cast<const uint8_t*>(x.data()),reinterpret_cast<const float*>(xs.data()),reinterpret_cast<const uint8_t*>(w.data()),reinterpret_cast<const float*>(ws.data()),x.size(0),w.size(0),x.size(1),ws.size(1),row_offset_,column_offset_);
  }
};
} // namespace infini::ops
