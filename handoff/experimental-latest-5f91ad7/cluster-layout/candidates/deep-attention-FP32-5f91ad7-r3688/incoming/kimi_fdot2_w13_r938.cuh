#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
typedef _Float16 HalfPair936 __attribute__((ext_vector_type(2)));
union HalfBits936 {uint32_t bits;HalfPair936 pair;};
// Convert only exactly representable finite BF16 values, preserving signed zero.
__device__ bool exact_half936(uint16_t b,uint16_t& h){
 unsigned e=(b>>7)&255,m=b&127,s=b&0x8000;
 if((b&0x7fff)==0){h=s;return true;}
 if(e>=113&&e<=142){h=s|(((b&0x7fff)-(112<<7))<<3);return true;}
 if(e>=110&&e<=112){h=s|((128+m)<<(e-110));return true;}
 if(e>=103&&e<=109){unsigned shift=110-e;if(((128+m)&((1u<<shift)-1))==0){h=s|((128+m)>>shift);return true;}}
 h=0;return false;
}
__device__ uint16_t raw_half936(unsigned nibble){unsigned m=nibble&7,s=(nibble&8)<<12;return s|(m==0?0:m==1?0x3800:((14+(m>>1))<<10)|((m&1)<<9));}
__device__ float bf936(uint16_t b){union{uint32_t u;float f;}x;x.u=uint32_t(b)<<16;return x.f;}
__device__ float raw936(unsigned x){const float values[8]={0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f};return x&8?-values[x&7]:values[x&7];}

namespace infini::ops::mxfp4_detail {
__global__ void prepareHalfR938(const uint16_t*x,uint32_t*halves,uint8_t*flags,size_t pairs){size_t p=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(p>=pairs)return;uint16_t a,b;bool ok=exact_half936(x[2*p],a);ok=exact_half936(x[2*p+1],b)&&ok;halves[p]=uint32_t(a)|(uint32_t(b)<<16);flags[p]=ok;}
template<class T> __device__ inline float pairR938(const T*x,size_t p,uint32_t halves,bool valid,uint8_t packed,uint8_t scale){
 if(valid&&scale>=64&&scale<=190){HalfBits936 a,w;a.bits=halves;w.bits=uint32_t(raw_half936(packed&15))|(uint32_t(raw_half936(packed>>4))<<16);float value=__builtin_amdgcn_fdot2(a.pair,w.pair,0.f,false);return ldexpf(value,int(scale)-127);}
 float low,high;mxfp4DecodePair(packed,scale,low,high);return mxfp4Load(x,2*p)*low+mxfp4Load(x,2*p+1)*high;
}
}
#pragma once
// Share weight decode across routes with the same expert, retaining each
// original 256-lane accumulation and binary reduction tree independently.
namespace infini::ops::mxfp4_detail {
template<int Tile>
__global__ void groupRoutesR938(const int32_t*ids,int32_t*groups,unsigned*count,size_t routes){
 size_t first=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(first>=routes)return;
 int32_t expert=ids[first];for(size_t p=0;p<first;++p)if(ids[p]==expert)return;
 unsigned matches=0;for(size_t p=first;p<routes;++p)matches+=ids[p]==expert;
 unsigned base=atomicAdd(count,(matches+Tile-1)/Tile),j=0;
 for(size_t p=first;p<routes;++p)if(ids[p]==expert){groups[(base+j/Tile)*Tile+j%Tile]=int32_t(p);++j;}
 for(;j%Tile;++j)groups[(base+j/Tile)*Tile+j%Tile]=-1;
}
template<class T,int Tile>
__global__ void groupedW13R938(T*activated,const T*input,const int32_t*ids,
 const uint8_t*packed,const uint8_t*scales,const int32_t*groups,const unsigned*count,size_t experts,size_t route_count,const uint32_t*half_input,const uint8_t*valid_input){
 constexpr size_t hidden=3584,intermediate=384,topk=16,pw=hidden/2,sw=hidden/32;
 size_t group=blockIdx.x/intermediate,i=blockIdx.x%intermediate;
 // Sparse routing may offer too little reuse to amortize extra accumulators.
 // All blocks see the same completed GPU grouping counter and choose one path.
 if(size_t(*count)*5>route_count*(Tile==4?3:4)){
  size_t route=group;if(route>=route_count)return;int32_t e=ids[route];
  if(e<0||size_t(e)>=experts){if(threadIdx.x==0)activated[route*intermediate+i]=mxfp4Store<T>(0.f);return;}
  size_t gr=size_t(e)*2*intermediate+i,ur=gr+intermediate;const T*x=input+(route/topk)*hidden;float sums[2]={};
  for(size_t p=threadIdx.x;p<pw;p+=blockDim.x){size_t index=(route/topk)*pw+p;uint32_t hx=half_input[index];bool valid=valid_input[index];sums[0]+=pairR938(x,p,hx,valid,packed[gr*pw+p],scales[gr*sw+p/16]);sums[1]+=pairR938(x,p,hx,valid,packed[ur*pw+p],scales[ur*sw+p/16]);}
  extern __shared__ float fallback_scratch[];mxfp4BlockReduce(sums,fallback_scratch);if(threadIdx.x==0)activated[route*intermediate+i]=mxfp4Store<T>(fusedMoeMxfp4Activate(sums[0],sums[1],size_t{2}));return;
 }
 if(group>=*count)return;
 int routes[Tile];for(int row=0;row<Tile;++row)routes[row]=groups[group*Tile+row];
 int32_t expert=ids[routes[0]];
 if(expert<0||size_t(expert)>=experts){if(threadIdx.x==0)for(int row=0;row<Tile;++row)if(routes[row]>=0)activated[size_t(routes[row])*intermediate+i]=mxfp4Store<T>(0.f);return;}
 size_t gr=size_t(expert)*2*intermediate+i,ur=gr+intermediate;float sums[2*Tile]={};
 for(size_t p=threadIdx.x;p<pw;p+=blockDim.x){
  
  #pragma unroll
  for(int row=0;row<Tile;++row)if(routes[row]>=0){size_t token=size_t(routes[row])/topk;const T*x=input+token*hidden;size_t index=token*pw+p;uint32_t hx=half_input[index];bool valid=valid_input[index];sums[2*row]+=pairR938(x,p,hx,valid,packed[gr*pw+p],scales[gr*sw+p/16]);sums[2*row+1]+=pairR938(x,p,hx,valid,packed[ur*pw+p],scales[ur*sw+p/16]);}
 }
 extern __shared__ float scratch[];mxfp4BlockReduce(sums,scratch);
 if(threadIdx.x==0)for(int row=0;row<Tile;++row)if(routes[row]>=0)activated[size_t(routes[row])*intermediate+i]=mxfp4Store<T>(fusedMoeMxfp4Activate(sums[2*row],sums[2*row+1],size_t{2}));
}
}
