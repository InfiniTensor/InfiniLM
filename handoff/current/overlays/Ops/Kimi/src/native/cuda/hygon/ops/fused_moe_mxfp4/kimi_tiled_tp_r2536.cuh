#pragma once
#include "kimi_reduce_wave_r1044.cuh"
namespace infini::ops::mxfp4_detail {
// Reuse each activation load over neighboring output channels. Preserve each
// channel's original256-lane product accumulation and binary reduction tree.
template<class T,int I,int Tile,int Wave>
__global__ void tiledW13TPR2536(T*activated,const T*input,const int32_t*ids,
 const uint8_t*packed,const uint8_t*scales,size_t experts,size_t routes){
 constexpr size_t H=3584,K=H/2,S=H/32;
 size_t route=blockIdx.x/(I/Tile),first=(blockIdx.x%(I/Tile))*Tile;
 if(route>=routes)return;
 int32_t expert=ids[route];
 if(expert<0||size_t(expert)>=experts){
  if(threadIdx.x==0)for(int j=0;j<Tile;++j)activated[route*I+first+j]=mxfp4Store<T>(0.f);
  return;
 }
 const T*x=input+(route/16)*H;float sums[2*Tile]={};
 #pragma unroll 1
 for(size_t p=threadIdx.x;p<K;p+=256){
  float a=mxfp4Load(x,2*p),b=mxfp4Load(x,2*p+1);
  #pragma unroll
  for(int j=0;j<Tile;++j){
   size_t gate=size_t(expert)*2*I+first+j,up=gate+I;
   float gl,gh,ul,uh;
   mxfp4DecodePair(packed[gate*K+p],scales[gate*S+p/16],gl,gh);
   mxfp4DecodePair(packed[up*K+p],scales[up*S+p/16],ul,uh);
   sums[2*j]+=a*gl+b*gh;sums[2*j+1]+=a*ul+b*uh;
  }
 }
 extern __shared__ float scratch[];
 if constexpr(Wave==0)mxfp4BlockReduce(sums,scratch);
 else reduceWaveR1044<2*Tile,Wave>(sums,scratch);
 if(threadIdx.x==0)for(int j=0;j<Tile;++j)
  activated[route*I+first+j]=mxfp4Store<T>(fusedMoeMxfp4Activate(sums[2*j],sums[2*j+1],size_t{2}));
}
}
