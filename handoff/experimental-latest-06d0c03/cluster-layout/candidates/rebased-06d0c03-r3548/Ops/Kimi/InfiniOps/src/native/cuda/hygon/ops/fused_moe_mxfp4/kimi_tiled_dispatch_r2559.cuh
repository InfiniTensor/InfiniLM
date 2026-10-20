#pragma once
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include "kimi_tiled_tp_r2536.cuh"

namespace infini::ops::mxfp4_detail {
inline std::array<std::atomic<unsigned long long>,8> tile_tp_calls_r2559{};
inline bool tile_tp_enabled_r2559(){
 // Fixed for the lifetime of a process; full-model A/B uses separate processes.
 static const bool enabled=[](){const char*p=std::getenv("INFINILM_MXFP4_TILE_TP_R2559");return p&&std::strcmp(p,"1")==0;}();
 return enabled;
}
template<int I,int Tile,class T,class Stream>
void tile_tp_launch_r2559(T*out,T*activated,const T*input,const int32_t*ids,
 const float*weights,const uint8_t*w13,const uint8_t*s13,const uint8_t*w2,
 const uint8_t*s2,const Mxfp4MoeGeometry&info,Stream stream){
 tiledW13TPR2536<T,I,Tile,64><<<unsigned(16*I/Tile),256,2*Tile*256*sizeof(float),stream>>>(activated,input,ids,w13,s13,info.num_experts,size_t{16});
 auto code=cudaGetLastError();if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));
 if constexpr(I==384)staticW2R326U<<<3584,256,0,stream>>>(out,activated,ids,weights,w2,s2,size_t{1},size_t{16},info.num_experts,size_t{3584},size_t{I},size_t{0});
 else fusedMoeMxfp4W2Routes16Kernel<<<3584,256,0,stream>>>(out,activated,ids,weights,w2,s2,size_t{1},size_t{16},info.num_experts,size_t{3584},size_t{I},size_t{0});
 code=cudaGetLastError();if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));
}
template<class T,class Stream>
bool try_tile_tp_r2559(T*out,T*activated,const T*input,const int32_t*ids,
 const float*weights,const uint8_t*w13,const uint8_t*s13,const uint8_t*w2,
 const uint8_t*s2,const Mxfp4MoeGeometry&info,Stream stream,bool parallel){
 if constexpr(!std::is_same_v<T,__nv_bfloat16>)return false;
 else{
  if(!parallel||info.num_tokens!=1||info.hidden_size!=3584||info.topk!=16||info.activation!=2||!tile_tp_enabled_r2559())return false;
  const auto i=info.intermediate_size;if(i!=64&&i!=96&&i!=192&&i!=384)return false;
  cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;
  auto capture_code=cudaStreamIsCapturing(static_cast<cudaStream_t>(stream),&capture);
  if(capture_code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(capture_code));
  if(capture!=cudaStreamCaptureStatusNone)throw std::logic_error("r2559 tiled TP candidate requires eager stream");
  int device=-1;if(cudaGetDevice(&device)!=cudaSuccess||device<0||device>=8)throw std::runtime_error("r2559 requires owned Hygon device");
  switch(i){
   case 64:tile_tp_launch_r2559<64,2>(out,activated,input,ids,weights,w13,s13,w2,s2,info,stream);break;
   case 96:tile_tp_launch_r2559<96,4>(out,activated,input,ids,weights,w13,s13,w2,s2,info,stream);break;
   case 192:tile_tp_launch_r2559<192,4>(out,activated,input,ids,weights,w13,s13,w2,s2,info,stream);break;
   case 384:tile_tp_launch_r2559<384,4>(out,activated,input,ids,weights,w13,s13,w2,s2,info,stream);break;
  }
  tile_tp_calls_r2559[size_t(device)].fetch_add(1,std::memory_order_relaxed);return true;
 }
}
}
extern "C" inline __attribute__((used,visibility("default"))) int modern_mxfp4_tile_tp_counts_r2559(unsigned long long*out,size_t count){
 if(!out||count!=8)return 1;for(size_t i=0;i<8;++i)out[i]=infini::ops::mxfp4_detail::tile_tp_calls_r2559[i].load(std::memory_order_relaxed);return 0;
}
