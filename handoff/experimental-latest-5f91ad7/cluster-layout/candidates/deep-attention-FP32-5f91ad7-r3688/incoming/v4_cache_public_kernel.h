#pragma once
#include "base/deepseek_v4_paged_cache.h"
#include <cuda_runtime.h>
namespace infini::ops::dsv4_cache_detail {
void Launch(size_t,size_t,size_t,size_t,size_t,void*,const void*,void*,void*,void*,cudaStream_t);
}
namespace infini::ops {
template<>class Operator<DeepseekV4PagedCache,Device::Type::kHygon>:public DeepseekV4PagedCache{
public:
 using DeepseekV4PagedCache::DeepseekV4PagedCache;
 bool capture_safe()const override{return false;}
 void operator()(Tensor cache,const Tensor slots,Tensor values,Tensor status,size_t block,size_t mode)const override{
  Invocation(cache,slots,values,status,block,mode);int device=-1;if(cache.device().type()!=Device::Type::kHygon||cudaGetDevice(&device)!=cudaSuccess||device!=cache.device().index())throw std::invalid_argument("V4 paged cache requires owning Hygon device");
  if(workspace_bytes_)moe_detail::Workspace(workspace_,workspace_size_in_bytes_,workspace_bytes_,{cache,slots,values,status});
  dsv4_cache_detail::Launch(mode,slots.size(0),cache.size(0),block,cache.size(1),cache.data(),slots.data(),values.data(),status.data(),workspace_,static_cast<cudaStream_t>(stream_));
 }
};
}
