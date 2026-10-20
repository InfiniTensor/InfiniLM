#pragma once
#include "base/deepseek_v4_compress.h"
#include <cuda_runtime.h>
namespace infini::ops::dsv4_compress_detail {
void Launch(void*,const void*,const void*,void*,bool,const void*,const void*,const void*,void*,void*,size_t,size_t,size_t,size_t,cudaStream_t);
}
namespace infini::ops {
template<>class Operator<DeepseekV4Compress,Device::Type::kHygon>:public DeepseekV4Compress{
public:
 using DeepseekV4Compress::DeepseekV4Compress;
 bool capture_safe()const override{return false;}
 void operator()(Tensor output,const Tensor score,const Tensor ape,Tensor state,const Tensor positions,const Tensor writes,const Tensor extras,Tensor status,size_t ratio)const override{
  Invocation(output,score,ape,state,positions,writes,extras,status,ratio);int device=-1;if(output.device().type()!=Device::Type::kHygon||cudaGetDevice(&device)!=cudaSuccess||device!=output.device().index())throw std::invalid_argument("V4 compression requires owning Hygon device");
  moe_detail::Workspace(workspace_,workspace_size_in_bytes_,workspace_bytes_,{output,score,ape,state,positions,writes,extras,status});
  dsv4_compress_detail::Launch(output.data(),score.data(),ape.data(),state.data(),state.dtype()==DataType::kFloat32,positions.data(),writes.data(),extras.data(),status.data(),workspace_,output.size(0),output.size(1),state.size(0),ratio,static_cast<cudaStream_t>(stream_));
 }
};
}
