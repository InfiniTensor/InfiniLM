#pragma once
#include "base/deepseek_v4_indexer_cache.h"
#include <cuda_runtime.h>
namespace infini::ops::dsv4_index_cache_detail{void Launch(void*,const void*,void*,void*,void*,void*,size_t,size_t,size_t,size_t,cudaStream_t);}
namespace infini::ops {
template<>class Operator<DeepseekV4IndexerCache,Device::Type::kHygon>:public DeepseekV4IndexerCache{
public:using DeepseekV4IndexerCache::DeepseekV4IndexerCache;bool capture_safe()const override{return false;}
 void operator()(Tensor cache,const Tensor slots,Tensor values,Tensor scales,Tensor status,size_t mode)const override{Invocation(cache,slots,values,scales,status,mode);int device=-1;if(cache.device().type()!=Device::Type::kHygon||cudaGetDevice(&device)!=cudaSuccess||device!=cache.device().index())throw std::invalid_argument("V4 index cache requires owner Hygon device");moe_detail::Workspace(workspace_,workspace_size_in_bytes_,workspace_bytes_,{cache,slots,values,scales,status});dsv4_index_cache_detail::Launch(cache.data(),slots.data(),values.data(),scales.data(),status.data(),workspace_,slots.size(0),cache.size(0),cache.size(1),mode,static_cast<cudaStream_t>(stream_));}
};
}
