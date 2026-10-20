#pragma once
// Experiment only: each physical lane retains the original 256-thread kernel's
// virtual accumulators. Reduce the high virtual-thread bits first, preserving
// the original binary tree before exchanging values inside a physical group.
namespace infini::ops::mxfp4_detail {
template<class T,int Width>
__global__ void W13VirtualWarp(T*activated,const T*input,const int32_t*ids,
 const uint8_t*packed,const uint8_t*scales,size_t routes,size_t topk,
 size_t experts,size_t hidden,size_t intermediate,size_t activation){
 constexpr int Virtual=256/Width,Groups=256/Width;
 size_t unit=size_t(blockIdx.x)*Groups+threadIdx.x/Width;
 size_t route=unit/intermediate,row=unit%intermediate;
 if(route>=routes)return;
 int expert=ids[route];unsigned lane=threadIdx.x%Width;
 if(expert<0||size_t(expert)>=experts){if(lane==0)activated[route*intermediate+row]=mxfp4Store<T>(0.f);return;}
 size_t pw=hidden/2,sw=hidden/32,gr=(size_t(expert)*2*intermediate+row),ur=gr+intermediate;
 const auto*x=input+(route/topk)*hidden;
 float gate[Virtual]={},up[Virtual]={};
 for(size_t base=lane;base<pw;base+=256){
  #pragma unroll
  for(int j=0;j<Virtual;++j){size_t p=base+size_t(j)*Width;if(p>=pw)continue;
   float gl,gh,ul,uh;mxfp4DecodePair(packed[gr*pw+p],scales[gr*sw+p/16],gl,gh);mxfp4DecodePair(packed[ur*pw+p],scales[ur*sw+p/16],ul,uh);
   float a=mxfp4Load(x,2*p),b=mxfp4Load(x,2*p+1);
   gate[j]+=a*gl+b*gh;up[j]+=a*ul+b*uh;
  }
 }
 #pragma unroll
 for(int stride=Virtual/2;stride;stride/=2){
  #pragma unroll
  for(int j=0;j<Virtual;++j)if(j<stride){gate[j]+=gate[j+stride];up[j]+=up[j+stride];}
 }
 float g=gate[0],u=up[0];
 #pragma unroll
 for(int shift=Width/2;shift;shift/=2){
#if defined(__HIPCC__)
  g+=__shfl_down(g,shift,Width);u+=__shfl_down(u,shift,Width);
#else
  g+=__shfl_down_sync(0xffffffffu,g,shift,Width);u+=__shfl_down_sync(0xffffffffu,u,shift,Width);
#endif
 }
 if(lane==0)activated[route*intermediate+row]=mxfp4Store<T>(fusedMoeMxfp4Activate(g,u,activation));
}
template<class T>void LaunchVirtualW13(T*out,T*scratch,const T*input,
 const int32_t*ids,const float*weights,const uint8_t*w13,const uint8_t*s13,
 const uint8_t*w2,const uint8_t*s2,size_t tokens,size_t experts,size_t hidden,
 size_t intermediate,size_t topk,size_t activation,int width,cudaStream_t stream){
 size_t work=tokens*topk*intermediate;
 if(width==64)W13VirtualWarp<T,64><<<unsigned((work+3)/4),256,0,stream>>>(scratch,input,ids,w13,s13,tokens*topk,topk,experts,hidden,intermediate,activation);
 else W13VirtualWarp<T,32><<<unsigned((work+7)/8),256,0,stream>>>(scratch,input,ids,w13,s13,tokens*topk,topk,experts,hidden,intermediate,activation);
 // Keep the already-qualified route-parallel W2 kernel exactly unchanged.
 fusedMoeMxfp4W2Routes16Kernel<<<unsigned(tokens*hidden),256,0,stream>>>(out,scratch,ids,weights,w2,s2,tokens,topk,experts,hidden,intermediate,size_t{0});
}
}
