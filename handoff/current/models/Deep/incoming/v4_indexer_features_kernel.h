#pragma once
#include "base/deepseek_v4_indexer_features.h"
#include <cuda_runtime.h>
namespace infini::ops::dsv4_index_features_detail {void Launch(void*,void*,const void*,const void*,const void*,void*,void*,size_t,size_t,bool,float,cudaStream_t);}
namespace infini::ops {
template<>class Operator<DeepseekV4IndexerFeatures,Device::Type::kHygon>:public DeepseekV4IndexerFeatures{
public:
 using DeepseekV4IndexerFeatures::DeepseekV4IndexerFeatures;
 bool capture_safe()const override{return false;}
 void operator()(Tensor output,Tensor scale,const Tensor input,const Tensor norm,const Tensor rope,Tensor status,bool normalize,float eps)const override{
  Invocation(output,scale,input,norm,rope,status,normalize,eps);int device=-1;if(output.device().type()!=Device::Type::kHygon||cudaGetDevice(&device)!=cudaSuccess||device!=output.device().index())throw std::invalid_argument("V4 indexer features require owner Hygon device");moe_detail::Workspace(workspace_,workspace_size_in_bytes_,workspace_bytes_,{output,scale,input,norm,rope,status});dsv4_index_features_detail::Launch(output.data(),scale.data(),input.data(),norm.data(),rope.data(),status.data(),workspace_,input.size(0),input.size(1),normalize,eps,static_cast<cudaStream_t>(stream_));
 }
};
}
