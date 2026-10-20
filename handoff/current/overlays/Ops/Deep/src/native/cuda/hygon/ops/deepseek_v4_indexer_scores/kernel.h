#pragma once
#include "base/deepseek_v4_indexer_scores.h"
#include <cuda_runtime.h>
namespace infini::ops::dsv4_index_scores_detail{void Launch(void*,const void*,const void*,const void*,const void*,const void*,void*,void*,size_t,size_t,size_t,cudaStream_t);}
namespace infini::ops {
template<>class Operator<DeepseekV4IndexerScores,Device::Type::kHygon>:public DeepseekV4IndexerScores{
public:using DeepseekV4IndexerScores::DeepseekV4IndexerScores;bool capture_safe()const override{return false;}
 void operator()(Tensor output,const Tensor query,const Tensor weights,const Tensor key,const Tensor scales,const Tensor visible,Tensor status)const override{Invocation(output,query,weights,key,scales,visible,status);int device=-1;if(output.device().type()!=Device::Type::kHygon||cudaGetDevice(&device)!=cudaSuccess||device!=output.device().index())throw std::invalid_argument("V4 index scores require owner Hygon device");moe_detail::Workspace(workspace_,workspace_size_in_bytes_,4,{output,query,weights,key,scales,visible,status});dsv4_index_scores_detail::Launch(output.data(),query.data(),weights.data(),key.data(),scales.data(),visible.data(),status.data(),workspace_,query.size(0),query.size(1),key.size(0),static_cast<cudaStream_t>(stream_));}
};
}
