#pragma once
#include "base/cast.h"
#include "dispatcher.h"
#include "native/cuda/hygon/caster.cuh"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
namespace infini::ops {
namespace hygon_cast {

// IEEE binary64 narrowing uses integer RNE, including subnormals and overflow.
// It avoids a binary32 intermediate for half/BF16 and respects signed zero.
__device__ inline uint32_t Narrow64(uint64_t u,unsigned p,unsigned ebits,int bias){
 const uint32_t sign=uint32_t(u>>63)<<(p+ebits),emax=(1u<<ebits)-1u;
 const unsigned e=unsigned((u>>52)&2047u);const uint64_t f=u&((uint64_t(1)<<52)-1);
 if(e==2047u)return sign|(emax<<p)|(f?(uint32_t(f>>(52-p))|(1u<<(p-1))):0u);
 if(!e)return sign;
 const int power=int(e)-1023,min_normal=1-bias,min_sub=min_normal-int(p);const uint64_t m=(uint64_t(1)<<52)|f;
 if(power<min_normal){
  if(power<min_sub-1)return sign;const unsigned shift=unsigned(52+min_sub-power);
  return sign|uint32_t((m+((uint64_t(1)<<(shift-1))-1)+((m>>shift)&1))>>shift);
 }
 int out_e=power+bias;if(out_e>=int(emax))return sign|(emax<<p);
 const unsigned shift=52-p;uint64_t rounded=(m+((uint64_t(1)<<(shift-1))-1)+((m>>shift)&1))>>shift;
 if(rounded==(uint64_t(1)<<(p+1))){rounded>>=1;++out_e;}
 if(out_e>=int(emax))return sign|(emax<<p);
 return sign|(uint32_t(out_e)<<p)|(uint32_t(rounded)&((1u<<p)-1u));
}
__device__ inline uint64_t Extend64(uint32_t u,unsigned p,unsigned ebits,int bias){
 const uint64_t sign=uint64_t(u>>(p+ebits))<<63;const uint32_t e=(u>>p)&((1u<<ebits)-1u),f=u&((1u<<p)-1u);
 if(e==((1u<<ebits)-1u))return sign|(uint64_t(2047)<<52)|(f?((uint64_t(f)<<(52-p))|(uint64_t(1)<<51)):0);
 if(e)return sign|(uint64_t(int(e)-bias+1023)<<52)|(uint64_t(f)<<(52-p));
 if(!f)return sign;const unsigned k=31u-unsigned(__clz(f));
 return sign|(uint64_t(1-bias-int(p)+int(k)+1023)<<52)|(uint64_t(f-(1u<<k))<<(52-k));
}
template<class Input,class Output> __global__ void Kernel(const Input *input,Output *out,size_t count){
 for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=size_t(gridDim.x)*blockDim.x){
  if constexpr(std::is_same_v<Input,double>){
   const uint64_t u=reinterpret_cast<const uint64_t*>(input)[i];
   if constexpr(std::is_same_v<Output,double>)reinterpret_cast<uint64_t*>(out)[i]=u;
   else if constexpr(std::is_same_v<Output,float>)reinterpret_cast<uint32_t*>(out)[i]=Narrow64(u,23,8,127);
   else if constexpr(std::is_same_v<Output,half>)reinterpret_cast<uint16_t*>(out)[i]=uint16_t(Narrow64(u,10,5,15));
   else reinterpret_cast<uint16_t*>(out)[i]=uint16_t(Narrow64(u,7,8,127));
  }else if constexpr(std::is_same_v<Output,double>){
   uint64_t u;
   if constexpr(std::is_same_v<Input,float>)u=Extend64(reinterpret_cast<const uint32_t*>(input)[i],23,8,127);
   else if constexpr(std::is_same_v<Input,half>)u=Extend64(reinterpret_cast<const uint16_t*>(input)[i],10,5,15);
   else u=Extend64(reinterpret_cast<const uint16_t*>(input)[i],7,8,127);
   reinterpret_cast<uint64_t*>(out)[i]=u;
  }else out[i]=Caster<Device::Type::kHygon>::template Cast<Output>(input[i]);
 }
}
inline bool SupportedType(DataType t){return t==DataType::kFloat64||t==DataType::kFloat32||t==DataType::kFloat16||t==DataType::kBFloat16;}
}
template<> class Operator<Cast,Device::Type::kHygon,0>:public Cast {
public:
 Operator(const Tensor input,Tensor out):Cast(input,out){if(!supports(input,out))throw std::invalid_argument("Unsupported Hygon contiguous floating Cast");}
 static bool supports(const Tensor input,Tensor out){
  if(input.device().type()!=Device::Type::kHygon||input.device()!=out.device()||input.shape()!=out.shape()
     ||!input.IsContiguous()||!out.IsContiguous()||!hygon_cast::SupportedType(input.dtype())||!hygon_cast::SupportedType(out.dtype()))return false;
  int current=-1;if(cudaGetDevice(&current)!=cudaSuccess||current!=out.device().index())return false;
  auto count=input.numel();if(!count)return true;
  const size_t maximum_width=(input.dtype()==DataType::kFloat64||out.dtype()==DataType::kFloat64)?8:4;
  if(count>std::numeric_limits<size_t>::max()/maximum_width)return false;
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
  using Types=List<DataType::kFloat64,DataType::kFloat32,DataType::kFloat16,DataType::kBFloat16>;
  DispatchFunc<Device::Type::kHygon,Types,Types>({input_dtype_,out_dtype_},[&](auto in_tag,auto out_tag){
   using In=typename decltype(in_tag)::type;using Out=typename decltype(out_tag)::type;
   size_t blocks=std::min<size_t>(65535,(output_size_+255)/256);
   hygon_cast::Kernel<In,Out><<<unsigned(blocks),256,0,static_cast<cudaStream_t>(stream_)>>>(static_cast<const In*>(input.data()),static_cast<Out*>(out.data()),output_size_);
  },"Hygon contiguous floating Cast");
  auto status=cudaGetLastError();if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
 }
};
}
