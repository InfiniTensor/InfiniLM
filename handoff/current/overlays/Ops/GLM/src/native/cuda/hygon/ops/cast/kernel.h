#pragma once
#include "base/cast.h"
#include "dispatcher.h"
#include "native/cuda/hygon/caster.cuh"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
namespace infini::ops {
namespace hygon_cast {
template<class Input,class Output> __global__ void Kernel(const Input *input,Output *out,size_t count){
 for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=size_t(gridDim.x)*blockDim.x)
  out[i]=Caster<Device::Type::kHygon>::template Cast<Output>(input[i]);
}
inline bool SupportedType(DataType t){return t==DataType::kFloat32||t==DataType::kFloat16||t==DataType::kBFloat16;}
}
template<> class Operator<Cast,Device::Type::kHygon,0>:public Cast {
public:
 Operator(const Tensor input,Tensor out):Cast(input,out){if(!supports(input,out))throw std::invalid_argument("Unsupported Hygon contiguous floating Cast");}
 static bool supports(const Tensor input,Tensor out){
  if(input.device().type()!=Device::Type::kHygon||input.device()!=out.device()||input.shape()!=out.shape()
     ||!input.IsContiguous()||!out.IsContiguous()||!hygon_cast::SupportedType(input.dtype())||!hygon_cast::SupportedType(out.dtype()))return false;
  int current=-1;if(cudaGetDevice(&current)!=cudaSuccess||current!=out.device().index())return false;
  auto count=input.numel();if(!count)return true;
  if(count>std::numeric_limits<size_t>::max()/4)return false;
  size_t in_bytes=count*kDataTypeToSize.at(input.dtype()),out_bytes=count*kDataTypeToSize.at(out.dtype());
  auto a=reinterpret_cast<uintptr_t>(input.data()),b=reinterpret_cast<uintptr_t>(out.data());
  if(a%kDataTypeToSize.at(input.dtype())||b%kDataTypeToSize.at(out.dtype()))return false;
  if(!a||!b||in_bytes>std::numeric_limits<uintptr_t>::max()-a||out_bytes>std::numeric_limits<uintptr_t>::max()-b)return false;
  bool overlap=a<b+out_bytes&&b<a+in_bytes;
  return !overlap||(a==b&&input.dtype()==out.dtype());
 }
 bool capture_safe()const override{return false;}
 void operator()(const Tensor input,Tensor out)const override{
  if(input.shape()!=input_shape_||out.shape()!=out_shape_||input.dtype()!=input_dtype_||out.dtype()!=out_dtype_||!supports(input,out))
   throw std::invalid_argument("Hygon Cast invocation metadata or alias mismatch");
  if(!output_size_||(input.data()==out.data()&&input.dtype()==out.dtype()))return;
  using Types=List<DataType::kFloat32,DataType::kFloat16,DataType::kBFloat16>;
  DispatchFunc<Device::Type::kHygon,Types,Types>({input_dtype_,out_dtype_},[&](auto in_tag,auto out_tag){
   using In=typename decltype(in_tag)::type;using Out=typename decltype(out_tag)::type;
   size_t blocks=std::min<size_t>(65535,(output_size_+255)/256);
   hygon_cast::Kernel<In,Out><<<unsigned(blocks),256,0,static_cast<cudaStream_t>(stream_)>>>(static_cast<const In*>(input.data()),static_cast<Out*>(out.data()),output_size_);
  },"Hygon contiguous floating Cast");
  auto status=cudaGetLastError();if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
 }
};
}
