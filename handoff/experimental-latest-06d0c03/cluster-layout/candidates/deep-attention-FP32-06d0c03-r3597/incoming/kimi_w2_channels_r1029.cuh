#pragma once
// Preserve each 32-lane expert dot and the sequential 0..15 weighted reduction.
// Half-warps in a block compute adjacent output channels, rather than experts.
namespace infini::ops::mxfp4_detail {
template<class T,int Channels>
__global__ void channelsW2R1029(T*out,const T*activated,const int32_t*ids,
 const float*weights,const uint8_t*packed,const uint8_t*scales,size_t tokens,size_t experts){
 constexpr size_t hidden=3584,intermediate=384,topk=16;
 const size_t lane=threadIdx.x%32,channel_group=threadIdx.x/32;
 const size_t index=size_t(blockIdx.x)*Channels+channel_group;
 const size_t token=index/hidden,channel=index%hidden;
 if(token>=tokens)return;
 float value=0.f;
 #pragma unroll 1
 for(size_t route=0;route<topk;++route){
  const int32_t expert=ids[token*topk+route];float sum=0.f;
  if(expert>=0&&size_t(expert)<experts){
   const size_t row=size_t(expert)*hidden+channel;
   #pragma unroll 6
   for(size_t p=lane;p<intermediate/2;p+=32){
    float low,high;mxfp4DecodePair(packed[row*(intermediate/2)+p],scales[row*(intermediate/32)+p/16],low,high);
    sum+=mxfp4Load(activated,(token*topk+route)*intermediate+2*p)*low
        +mxfp4Load(activated,(token*topk+route)*intermediate+2*p+1)*high;
   }
  }
  for(int shift=16;shift>0;shift/=2){
#if defined(__HIPCC__)
   sum+=__shfl_down(sum,shift,32);
#else
   sum+=__shfl_down_sync(0xffffffffu,sum,shift,32);
#endif
  }
  if(lane==0&&expert>=0&&size_t(expert)<experts)value+=weights[token*topk+route]*sum;
 }
 if(lane==0)out[index]=mxfp4Store<T>(value);
}
}
