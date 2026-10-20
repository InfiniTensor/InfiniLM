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
    fp8_block_detail::Linear<T><<<dim3(weight.size(0),x.size(0)),256,0,reinterpret_cast<cudaStream_t>(stream_)>>>(
      reinterpret_cast<T*>(output.data()),reinterpret_cast<const uint8_t*>(x.data()),reinterpret_cast<const float*>(x_scale.data()),
      reinterpret_cast<const uint8_t*>(weight.data()),reinterpret_cast<const float*>(weight_scale.data()),
      x.size(0),weight.size(0),x.size(1),weight_scale.size(1),row_offset_,column_offset_);
  }
};
} // namespace infini::ops
