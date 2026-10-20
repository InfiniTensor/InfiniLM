#pragma once
#include "kimi_mxfp4_adaptive_w13_r617.cuh"
namespace infini::ops::mxfp4_detail {
// Keep each route's 256-lane arithmetic/reduction tree. Only scheduling changes:
// a bounded grid iterates GPU-formed expert groups or the original route fallback.
template<class T,int Tile>
__global__ void persistentW13R704(T*activated,const T*input,const int32_t*ids,
 const uint8_t*packed,const uint8_t*scales,const int32_t*groups,const unsigned*count,
 size_t experts,size_t route_count){
 constexpr size_t hidden=3584,intermediate=384,topk=16,pw=hidden/2,sw=hidden/32;
 const bool fallback=size_t(*count)*5>route_count*(Tile==4?3:4);
 const size_t total=fallback?route_count:size_t(*count),stride=gridDim.x/intermediate;
 const size_t i=blockIdx.x%intermediate;
 for(size_t group=blockIdx.x/intermediate;group<total;group+=stride){
  if(fallback){
   const size_t route=group;int32_t e=ids[route];
   if(e<0||size_t(e)>=experts){if(threadIdx.x==0)activated[route*intermediate+i]=mxfp4Store<T>(0.f);__syncthreads();continue;}
   size_t gr=size_t(e)*2*intermediate+i,ur=gr+intermediate;const T*x=input+(route/topk)*hidden;float sums[2]={};
   for(size_t p=threadIdx.x;p<pw;p+=blockDim.x){float gl,gh,ul,uh;mxfp4DecodePair(packed[gr*pw+p],scales[gr*sw+p/16],gl,gh);mxfp4DecodePair(packed[ur*pw+p],scales[ur*sw+p/16],ul,uh);float a=mxfp4Load(x,2*p),b=mxfp4Load(x,2*p+1);sums[0]+=a*gl+b*gh;sums[1]+=a*ul+b*uh;}
   extern __shared__ float scratch[];mxfp4BlockReduce(sums,scratch);
   if(threadIdx.x==0)activated[route*intermediate+i]=mxfp4Store<T>(fusedMoeMxfp4Activate(sums[0],sums[1],size_t{2}));
   __syncthreads();continue;
  }
  int routes[Tile];for(int row=0;row<Tile;++row)routes[row]=groups[group*Tile+row];
  int32_t expert=ids[routes[0]];
  if(expert<0||size_t(expert)>=experts){if(threadIdx.x==0)for(int row=0;row<Tile;++row)if(routes[row]>=0)activated[size_t(routes[row])*intermediate+i]=mxfp4Store<T>(0.f);__syncthreads();continue;}
  size_t gr=size_t(expert)*2*intermediate+i,ur=gr+intermediate;float sums[2*Tile]={};
  for(size_t p=threadIdx.x;p<pw;p+=blockDim.x){float gl,gh,ul,uh;mxfp4DecodePair(packed[gr*pw+p],scales[gr*sw+p/16],gl,gh);mxfp4DecodePair(packed[ur*pw+p],scales[ur*sw+p/16],ul,uh);
   #pragma unroll
   for(int row=0;row<Tile;++row)if(routes[row]>=0){const T*x=input+(size_t(routes[row])/topk)*hidden;float a=mxfp4Load(x,p*2),b=mxfp4Load(x,p*2+1);sums[2*row]+=a*gl+b*gh;sums[2*row+1]+=a*ul+b*uh;}
  }
  extern __shared__ float scratch[];mxfp4BlockReduce(sums,scratch);
  if(threadIdx.x==0)for(int row=0;row<Tile;++row)if(routes[row]>=0)activated[size_t(routes[row])*intermediate+i]=mxfp4Store<T>(fusedMoeMxfp4Activate(sums[2*row],sums[2*row+1],size_t{2}));
  __syncthreads();
 }
}
}
