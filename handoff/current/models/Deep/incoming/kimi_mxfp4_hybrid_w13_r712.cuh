#pragma once
#include "kimi_mxfp4_adaptive_w13_r617.cuh"
namespace infini::ops::mxfp4_detail {
template<class T,int OutputTile>
__global__ void hybridW13R712(T*activated,const T*input,const int32_t*ids,
 const uint8_t*packed,const uint8_t*scales,const int32_t*groups,const unsigned*count,
 size_t experts,size_t route_count){
 constexpr size_t hidden=3584,intermediate=384,topk=16,pw=hidden/2,sw=hidden/32,GroupTile=4;
 const bool low_reuse=size_t(*count)*5>route_count*3;
 const size_t total=low_reuse?route_count*(intermediate/OutputTile):size_t(*count)*intermediate;
 for(size_t work=blockIdx.x;work<total;work+=gridDim.x){
  if(low_reuse){
   const size_t route=work/(intermediate/OutputTile),base=(work%(intermediate/OutputTile))*OutputTile;int32_t expert=ids[route];
   if(expert<0||size_t(expert)>=experts){if(threadIdx.x==0)for(int j=0;j<OutputTile;++j)activated[route*intermediate+base+j]=mxfp4Store<T>(0.f);__syncthreads();continue;}
   const T*x=input+(route/topk)*hidden;float sums[2*OutputTile]={};
   for(size_t p=threadIdx.x;p<pw;p+=blockDim.x){float a=mxfp4Load(x,2*p),b=mxfp4Load(x,2*p+1);
    #pragma unroll
    for(int j=0;j<OutputTile;++j){size_t gr=size_t(expert)*2*intermediate+base+j,ur=gr+intermediate;float gl,gh,ul,uh;mxfp4DecodePair(packed[gr*pw+p],scales[gr*sw+p/16],gl,gh);mxfp4DecodePair(packed[ur*pw+p],scales[ur*sw+p/16],ul,uh);sums[2*j]+=a*gl+b*gh;sums[2*j+1]+=a*ul+b*uh;}
   }
   extern __shared__ float scratch[];mxfp4BlockReduce(sums,scratch);
   if(threadIdx.x==0)for(int j=0;j<OutputTile;++j)activated[route*intermediate+base+j]=mxfp4Store<T>(fusedMoeMxfp4Activate(sums[2*j],sums[2*j+1],size_t{2}));
   __syncthreads();continue;
  }
  const size_t group=work/intermediate,i=work%intermediate;int routes[GroupTile];for(int j=0;j<GroupTile;++j)routes[j]=groups[group*GroupTile+j];int32_t expert=ids[routes[0]];
  if(expert<0||size_t(expert)>=experts){if(threadIdx.x==0)for(int j=0;j<GroupTile;++j)if(routes[j]>=0)activated[size_t(routes[j])*intermediate+i]=mxfp4Store<T>(0.f);__syncthreads();continue;}
  size_t gr=size_t(expert)*2*intermediate+i,ur=gr+intermediate;float sums[2*GroupTile]={};
  for(size_t p=threadIdx.x;p<pw;p+=blockDim.x){float gl,gh,ul,uh;mxfp4DecodePair(packed[gr*pw+p],scales[gr*sw+p/16],gl,gh);mxfp4DecodePair(packed[ur*pw+p],scales[ur*sw+p/16],ul,uh);
   #pragma unroll
   for(int j=0;j<GroupTile;++j)if(routes[j]>=0){const T*x=input+(size_t(routes[j])/topk)*hidden;float a=mxfp4Load(x,p*2),b=mxfp4Load(x,p*2+1);sums[2*j]+=a*gl+b*gh;sums[2*j+1]+=a*ul+b*uh;}
  }
  extern __shared__ float scratch[];mxfp4BlockReduce(sums,scratch);
  if(threadIdx.x==0)for(int j=0;j<GroupTile;++j)if(routes[j]>=0)activated[size_t(routes[j])*intermediate+i]=mxfp4Store<T>(fusedMoeMxfp4Activate(sums[2*j],sums[2*j+1],size_t{2}));
  __syncthreads();
 }
}
}
