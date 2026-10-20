#pragma once
#include <algorithm>
#include <atomic>
#include <stdexcept>
#include <type_traits>
#include "kimi_mxfp4_static_r326.cuh"

namespace infini::ops::mxfp4_detail {
inline std::atomic<unsigned long long> staticW2CallsR330[8]{};
template<class T,class Stream>
void launchFusedMoeMxfp4R330(T*out,T*activated,const T*input,const int32_t*ids,
 const float*weights,const uint8_t*w13,const uint8_t*s13,const uint8_t*w2,
 const uint8_t*s2,const Mxfp4MoeGeometry&info,Stream stream,bool parallel){
 if constexpr(std::is_same_v<T,__nv_bfloat16>){
  if(parallel&&info.num_tokens>0&&info.num_tokens<=64&&info.hidden_size==3584&&
     info.intermediate_size==384&&info.topk==16&&info.activation==2){
   int device=-1;
   if(cudaGetDevice(&device)!=cudaSuccess||device<0||device>=8)
    throw std::runtime_error("R330 requires an owned local BW device0..7");
   constexpr size_t max_blocks=std::numeric_limits<uint32_t>::max()/256;
   const size_t first=info.routeCount()*info.intermediate_size;
   for(size_t offset=0;offset<first;offset+=max_blocks){
    size_t blocks=std::min(max_blocks,first-offset);
    fusedMoeMxfp4W13Kernel<<<unsigned(blocks),256,2048,stream>>>(activated,input,ids,w13,s13,
     info.routeCount(),info.topk,info.num_experts,info.hidden_size,info.intermediate_size,info.activation,offset);
   }
   const size_t second=info.num_tokens*info.hidden_size;
   for(size_t offset=0;offset<second;offset+=max_blocks){
    size_t blocks=std::min(max_blocks,second-offset);
    staticW2R326U<<<unsigned(blocks),256,0,stream>>>(out,activated,ids,weights,w2,s2,
     info.num_tokens,info.topk,info.num_experts,info.hidden_size,info.intermediate_size,offset);
   }
   staticW2CallsR330[device].fetch_add(1,std::memory_order_relaxed);
   return;
  }
 }
 launchFusedMoeMxfp4(out,activated,input,ids,weights,w13,s13,w2,s2,info,stream,parallel);
}
}
extern "C" inline __attribute__((used,visibility("default"))) int
modern_mxfp4_static_r330_counts(unsigned long long*out,size_t count){
 if(!out||count!=8)return 1;
 for(size_t i=0;i<8;++i)out[i]=infini::ops::mxfp4_detail::staticW2CallsR330[i].load(std::memory_order_relaxed);
 return 0;
}
