#pragma once
#include <mma.h>
#include <algorithm>
namespace infini::ops::mxfp4_detail {
template<class T,int Tile>
__global__ void hybridFallback909(T*activated,const T*input,const int32_t*ids,
 const uint8_t*packed,const uint8_t*scales,const int32_t*groups,const unsigned*count,size_t experts,size_t route_count,const unsigned*tc_count){
 if(size_t(*tc_count)*2<=route_count)return;
 constexpr size_t hidden=3584,intermediate=384,topk=16,pw=hidden/2,sw=hidden/32;
 size_t group=blockIdx.x/intermediate,i=blockIdx.x%intermediate;
 // Sparse routing may offer too little reuse to amortize extra accumulators.
 // All blocks see the same completed GPU grouping counter and choose one path.
 if(size_t(*count)*5>route_count*(Tile==4?3:4)){
  size_t route=group;if(route>=route_count)return;int32_t e=ids[route];
  if(e<0||size_t(e)>=experts){if(threadIdx.x==0)activated[route*intermediate+i]=mxfp4Store<T>(0.f);return;}
  size_t gr=size_t(e)*2*intermediate+i,ur=gr+intermediate;const T*x=input+(route/topk)*hidden;float sums[2]={};
  for(size_t p=threadIdx.x;p<pw;p+=blockDim.x){float gl,gh,ul,uh;mxfp4DecodePair(packed[gr*pw+p],scales[gr*sw+p/16],gl,gh);mxfp4DecodePair(packed[ur*pw+p],scales[ur*sw+p/16],ul,uh);float a=mxfp4Load(x,2*p),b=mxfp4Load(x,2*p+1);sums[0]+=a*gl+b*gh;sums[1]+=a*ul+b*uh;}
  extern __shared__ float fallback_scratch[];mxfp4BlockReduce(sums,fallback_scratch);if(threadIdx.x==0)activated[route*intermediate+i]=mxfp4Store<T>(fusedMoeMxfp4Activate(sums[0],sums[1],size_t{2}));return;
 }
 if(group>=*count)return;
 int routes[Tile];for(int row=0;row<Tile;++row)routes[row]=groups[group*Tile+row];
 int32_t expert=ids[routes[0]];
 if(expert<0||size_t(expert)>=experts){if(threadIdx.x==0)for(int row=0;row<Tile;++row)if(routes[row]>=0)activated[size_t(routes[row])*intermediate+i]=mxfp4Store<T>(0.f);return;}
 size_t gr=size_t(expert)*2*intermediate+i,ur=gr+intermediate;float sums[2*Tile]={};
 for(size_t p=threadIdx.x;p<pw;p+=blockDim.x){
  float gl,gh,ul,uh;mxfp4DecodePair(packed[gr*pw+p],scales[gr*sw+p/16],gl,gh);mxfp4DecodePair(packed[ur*pw+p],scales[ur*sw+p/16],ul,uh);
  #pragma unroll
  for(int row=0;row<Tile;++row)if(routes[row]>=0){const T*x=input+(size_t(routes[row])/topk)*hidden;float a=mxfp4Load(x,p*2),b=mxfp4Load(x,p*2+1);sums[2*row]+=a*gl+b*gh;sums[2*row+1]+=a*ul+b*uh;}
 }
 extern __shared__ float scratch[];mxfp4BlockReduce(sums,scratch);
 if(threadIdx.x==0)for(int row=0;row<Tile;++row)if(routes[row]>=0)activated[size_t(routes[row])*intermediate+i]=mxfp4Store<T>(fusedMoeMxfp4Activate(sums[2*row],sums[2*row+1],size_t{2}));
}template<class T,int Tile>
__global__ void persistentFallback909(T*activated,const T*input,const int32_t*ids,
 const uint8_t*packed,const uint8_t*scales,const int32_t*groups,const unsigned*count,size_t experts,size_t route_count,const unsigned*tc_count){
 if(size_t(*tc_count)*2<=route_count)return;
 constexpr size_t hidden=3584,intermediate=384,topk=16,pw=hidden/2,sw=hidden/32;
 for(size_t job=blockIdx.x;job<route_count*intermediate;job+=gridDim.x){size_t group=job/intermediate,i=job%intermediate;
 // Sparse routing may offer too little reuse to amortize extra accumulators.
 // All blocks see the same completed GPU grouping counter and choose one path.
 if(size_t(*count)*5>route_count*(Tile==4?3:4)){
  size_t route=group;if(route>=route_count){__syncthreads();continue;}int32_t e=ids[route];
  if(e<0||size_t(e)>=experts){if(threadIdx.x==0)activated[route*intermediate+i]=mxfp4Store<T>(0.f);{__syncthreads();continue;}}
  size_t gr=size_t(e)*2*intermediate+i,ur=gr+intermediate;const T*x=input+(route/topk)*hidden;float sums[2]={};
  for(size_t p=threadIdx.x;p<pw;p+=blockDim.x){float gl,gh,ul,uh;mxfp4DecodePair(packed[gr*pw+p],scales[gr*sw+p/16],gl,gh);mxfp4DecodePair(packed[ur*pw+p],scales[ur*sw+p/16],ul,uh);float a=mxfp4Load(x,2*p),b=mxfp4Load(x,2*p+1);sums[0]+=a*gl+b*gh;sums[1]+=a*ul+b*uh;}
  extern __shared__ float fallback_scratch[];mxfp4BlockReduce(sums,fallback_scratch);if(threadIdx.x==0)activated[route*intermediate+i]=mxfp4Store<T>(fusedMoeMxfp4Activate(sums[0],sums[1],size_t{2}));{__syncthreads();continue;}
 }
 if(group>=*count){__syncthreads();continue;}
 int routes[Tile];for(int row=0;row<Tile;++row)routes[row]=groups[group*Tile+row];
 int32_t expert=ids[routes[0]];
 if(expert<0||size_t(expert)>=experts){if(threadIdx.x==0)for(int row=0;row<Tile;++row)if(routes[row]>=0)activated[size_t(routes[row])*intermediate+i]=mxfp4Store<T>(0.f);{__syncthreads();continue;}}
 size_t gr=size_t(expert)*2*intermediate+i,ur=gr+intermediate;float sums[2*Tile]={};
 for(size_t p=threadIdx.x;p<pw;p+=blockDim.x){
  float gl,gh,ul,uh;mxfp4DecodePair(packed[gr*pw+p],scales[gr*sw+p/16],gl,gh);mxfp4DecodePair(packed[ur*pw+p],scales[ur*sw+p/16],ul,uh);
  #pragma unroll
  for(int row=0;row<Tile;++row)if(routes[row]>=0){const T*x=input+(size_t(routes[row])/topk)*hidden;float a=mxfp4Load(x,p*2),b=mxfp4Load(x,p*2+1);sums[2*row]+=a*gl+b*gh;sums[2*row+1]+=a*ul+b*uh;}
 }
 extern __shared__ float scratch[];mxfp4BlockReduce(sums,scratch);
 if(threadIdx.x==0)for(int row=0;row<Tile;++row)if(routes[row]>=0)activated[size_t(routes[row])*intermediate+i]=mxfp4Store<T>(fusedMoeMxfp4Activate(sums[2*row],sums[2*row+1],size_t{2}));
 __syncthreads();
 }
}
}
namespace {
namespace wm909=nvcuda::wmma;
using B909=__nv_bfloat16;
using A909=wm909::fragment<wm909::matrix_a,16,16,16,B909,wm909::row_major>;
using W909=wm909::fragment<wm909::matrix_b,16,16,16,B909,wm909::col_major>;
using C909=wm909::fragment<wm909::accumulator,16,16,16,float>;
__device__ __forceinline__ uint16_t fp4bits909(unsigned code,unsigned scale){
 unsigned mag=code&7,sign=(code&8)<<12;if(!mag)return uint16_t(sign);
 int exp=int(scale)+int(mag/2)-1;unsigned mant=mag>=2&&(mag&1)?64:0;
 if(exp>0&&exp<255)return uint16_t(sign|(unsigned(exp)<<7)|mant);
 return __bfloat16_as_ushort(__float2bfloat16_rn(ldexpf(mxfp4DecodeE2M1(uint8_t(code)),int(scale)-127)));
}
__global__ void pair909(uint32_t*fast,uint32_t*old){
 unsigned p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=65536)return;unsigned byte=p&255,scale=p>>8;float a,b;mxfp4DecodePair(uint8_t(byte),uint8_t(scale),a,b);
 fast[p]=uint32_t(fp4bits909(byte&15,scale))|(uint32_t(fp4bits909(byte>>4,scale))<<16);old[p]=uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(a)))|(uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(b)))<<16);
}
__global__ void groupOriginal909(const int32_t*ids,int32_t*groups,unsigned*count,unsigned routes){
 __shared__ unsigned counts[1024];unsigned first=threadIdx.x,matches=0;
 if(first<routes){bool leader=true;for(unsigned p=0;p<first;++p)if(ids[p]==ids[first])leader=false;if(leader)for(unsigned p=first;p<routes;++p)matches+=ids[p]==ids[first];}
 counts[first]=(matches+15)/16;__syncthreads();unsigned base=0;for(unsigned p=0;p<first;++p)base+=counts[p];if(first==blockDim.x-1)*count=base+counts[first];
 if(matches){unsigned j=0;for(unsigned p=first;p<routes;++p)if(ids[p]==ids[first]){groups[(base+j/16)*16+j%16]=int32_t(p);++j;}for(;j%16;++j)groups[(base+j/16)*16+j%16]=-1;}
}
__global__ void group909(const int32_t*ids,int32_t*groups,unsigned*count,unsigned routes){
 __shared__ unsigned counts[1024];
 for(unsigned first=threadIdx.x;first<routes;first+=blockDim.x){unsigned matches=0;bool leader=true;for(unsigned p=0;p<first;++p)if(ids[p]==ids[first])leader=false;if(leader)for(unsigned p=first;p<routes;++p)matches+=ids[p]==ids[first];counts[first]=(matches+15)/16;}
 __syncthreads();
 if(threadIdx.x==0){unsigned total=0;for(unsigned p=0;p<routes;++p)total+=counts[p];*count=total;}
 for(unsigned first=threadIdx.x;first<routes;first+=blockDim.x){if(!counts[first])continue;unsigned base=0;for(unsigned p=0;p<first;++p)base+=counts[p];unsigned j=0;for(unsigned p=first;p<routes;++p)if(ids[p]==ids[first]){groups[(base+j/16)*16+j%16]=int32_t(p);++j;}for(;j%16;++j)groups[(base+j/16)*16+j%16]=-1;}
}
__device__ __forceinline__ B909 weight909(const uint8_t*p,const uint8_t*s,size_t row,unsigned k){
 unsigned byte=p[row*1792+k/2],scale=s[row*112+k/32];return __ushort_as_bfloat16(fp4bits909((byte>>((k&1)*4))&15,scale));
}
// Direct fragments avoid decoded-bank allocation and host route-counter waits.
__global__ void direct909(B909*out,const B909*x,const int32_t*ids,const uint8_t*p,const uint8_t*s,const int32_t*groups,const unsigned*count,size_t experts){
 unsigned group=blockIdx.y;if(group>=*count)return;unsigned lane=threadIdx.x%64,wave=threadIdx.x/64,nbase=blockIdx.x*64+wave*16;
 int route=groups[group*16+(lane&15)],expert=ids[groups[group*16]];C909 gate,up;wm909::fill_fragment(gate,0.f);wm909::fill_fragment(up,0.f);
 if(expert>=0&&size_t(expert)<experts){
  for(unsigned k=0;k<3584;k+=16){A909 a;W909 g,u;
   #pragma unroll
   for(unsigned j=0;j<4;++j){unsigned kk=k+(lane/16)*4+j;a.x[j]=route<0?__float2bfloat16_rn(0.f):x[size_t(route/16)*3584+kk];size_t row=size_t(expert)*768+nbase+(lane&15);g.x[j]=weight909(p,s,row,kk);u.x[j]=weight909(p,s,row+384,kk);}
   wm909::mma_sync(gate,a,g,gate);wm909::mma_sync(up,a,u,up);
  }
 }
 if(route>=0){
  #pragma unroll
  for(unsigned j=0;j<4;++j){unsigned n=nbase+lane/16+j*4;float v=infini::ops::mxfp4_detail::fusedMoeMxfp4Activate(gate.x[j],up.x[j],size_t{2});out[size_t(route)*384+n]=__float2bfloat16_rn(v);}
 }
}
// Padding changes only LDS row pitch; identical coalesced global loads,
// FP4 decode, logical fragments, accumulation order and BF16 outputs.
// Modes2/3/4/5 use pitch64/66/68/72; fixed256-thread grouping inherited853.
template<unsigned Pitch> __global__ void shared909(B909*out,const B909*x,const int32_t*ids,const uint8_t*p,const uint8_t*s,const int32_t*groups,const unsigned*count,size_t experts){
 unsigned group=blockIdx.y;if(group>=*count)return;
 __shared__ __align__(16) B909 xs[16*Pitch],gs[64*Pitch],us[64*Pitch];
 unsigned t=threadIdx.x,wave=t/64,nbase=blockIdx.x*64;int expert=ids[groups[group*16]];C909 gate,up;wm909::fill_fragment(gate,0.f);wm909::fill_fragment(up,0.f);
 if(expert>=0&&size_t(expert)<experts){
  for(unsigned k=0;k<3584;k+=64){
   for(unsigned q=t;q<16*64;q+=256){int route=groups[group*16+q/64];xs[(q/64)*Pitch+q%64]=route<0?__float2bfloat16_rn(0.f):x[size_t(route/16)*3584+k+q%64];}
   for(unsigned q=t;q<64*32;q+=256){unsigned n=q/32,col=q%32;size_t row=size_t(expert)*768+nbase+n;unsigned kk=k/2+col;unsigned g=p[row*1792+kk],u=p[(row+384)*1792+kk],sg=s[row*112+kk/16],su=s[(row+384)*112+kk/16];reinterpret_cast<uint32_t*>(gs)[n*(Pitch/2)+col]=uint32_t(fp4bits909(g&15,sg))|(uint32_t(fp4bits909(g>>4,sg))<<16);reinterpret_cast<uint32_t*>(us)[n*(Pitch/2)+col]=uint32_t(fp4bits909(u&15,su))|(uint32_t(fp4bits909(u>>4,su))<<16);}
   __syncthreads();
   #pragma unroll
   for(unsigned kk=0;kk<64;kk+=16){A909 a;W909 g,u;wm909::load_matrix_sync(a,xs+kk,Pitch);wm909::load_matrix_sync(g,gs+wave*16*Pitch+kk,Pitch);wm909::load_matrix_sync(u,us+wave*16*Pitch+kk,Pitch);wm909::mma_sync(gate,a,g,gate);wm909::mma_sync(up,a,u,up);}
   __syncthreads();
  }
 }
 int route=groups[group*16+(t%64&15)];if(route>=0){
  #pragma unroll
  for(unsigned j=0;j<4;++j){unsigned n=nbase+wave*16+(t%64)/16+j*4;float v=infini::ops::mxfp4_detail::fusedMoeMxfp4Activate(gate.x[j],up.x[j],size_t{2});out[size_t(route)*384+n]=__float2bfloat16_rn(v);}
 }
}
template<unsigned Pitch> __global__ void hybridTC909(B909*out,const B909*x,const int32_t*ids,const uint8_t*p,const uint8_t*s,const int32_t*groups,const unsigned*count,size_t experts,size_t route_count){
 unsigned group=blockIdx.y;if(group>=*count||size_t(*count)*2>route_count)return;
 __shared__ __align__(16) B909 xs[16*Pitch],gs[64*Pitch],us[64*Pitch];
 unsigned t=threadIdx.x,wave=t/64,nbase=blockIdx.x*64;int expert=ids[groups[group*16]];C909 gate,up;wm909::fill_fragment(gate,0.f);wm909::fill_fragment(up,0.f);
 if(expert>=0&&size_t(expert)<experts){
  for(unsigned k=0;k<3584;k+=64){
   for(unsigned q=t;q<16*64;q+=256){int route=groups[group*16+q/64];xs[(q/64)*Pitch+q%64]=route<0?__float2bfloat16_rn(0.f):x[size_t(route/16)*3584+k+q%64];}
   for(unsigned q=t;q<64*32;q+=256){unsigned n=q/32,col=q%32;size_t row=size_t(expert)*768+nbase+n;unsigned kk=k/2+col;unsigned g=p[row*1792+kk],u=p[(row+384)*1792+kk],sg=s[row*112+kk/16],su=s[(row+384)*112+kk/16];reinterpret_cast<uint32_t*>(gs)[n*(Pitch/2)+col]=uint32_t(fp4bits909(g&15,sg))|(uint32_t(fp4bits909(g>>4,sg))<<16);reinterpret_cast<uint32_t*>(us)[n*(Pitch/2)+col]=uint32_t(fp4bits909(u&15,su))|(uint32_t(fp4bits909(u>>4,su))<<16);}
   __syncthreads();
   #pragma unroll
   for(unsigned kk=0;kk<64;kk+=16){A909 a;W909 g,u;wm909::load_matrix_sync(a,xs+kk,Pitch);wm909::load_matrix_sync(g,gs+wave*16*Pitch+kk,Pitch);wm909::load_matrix_sync(u,us+wave*16*Pitch+kk,Pitch);wm909::mma_sync(gate,a,g,gate);wm909::mma_sync(up,a,u,up);}
   __syncthreads();
  }
 }
 int route=groups[group*16+(t%64&15)];if(route>=0){
  #pragma unroll
  for(unsigned j=0;j<4;++j){unsigned n=nbase+wave*16+(t%64)/16+j*4;float v=infini::ops::mxfp4_detail::fusedMoeMxfp4Activate(gate.x[j],up.x[j],size_t{2});out[size_t(route)*384+n]=__float2bfloat16_rn(v);}
 }
}
template<unsigned Pitch> __global__ void persistentTC909(B909*out,const B909*x,const int32_t*ids,const uint8_t*p,const uint8_t*s,const int32_t*groups,const unsigned*count,size_t experts,size_t route_count){
 if(size_t(*count)*2>route_count)return;
 for(unsigned group=blockIdx.y;group<*count;group+=gridDim.y){
 __shared__ __align__(16) B909 xs[16*Pitch],gs[64*Pitch],us[64*Pitch];
 unsigned t=threadIdx.x,wave=t/64,nbase=blockIdx.x*64;int expert=ids[groups[group*16]];C909 gate,up;wm909::fill_fragment(gate,0.f);wm909::fill_fragment(up,0.f);
 if(expert>=0&&size_t(expert)<experts){
  for(unsigned k=0;k<3584;k+=64){
   for(unsigned q=t;q<16*64;q+=256){int route=groups[group*16+q/64];xs[(q/64)*Pitch+q%64]=route<0?__float2bfloat16_rn(0.f):x[size_t(route/16)*3584+k+q%64];}
   for(unsigned q=t;q<64*32;q+=256){unsigned n=q/32,col=q%32;size_t row=size_t(expert)*768+nbase+n;unsigned kk=k/2+col;unsigned g=p[row*1792+kk],u=p[(row+384)*1792+kk],sg=s[row*112+kk/16],su=s[(row+384)*112+kk/16];reinterpret_cast<uint32_t*>(gs)[n*(Pitch/2)+col]=uint32_t(fp4bits909(g&15,sg))|(uint32_t(fp4bits909(g>>4,sg))<<16);reinterpret_cast<uint32_t*>(us)[n*(Pitch/2)+col]=uint32_t(fp4bits909(u&15,su))|(uint32_t(fp4bits909(u>>4,su))<<16);}
   __syncthreads();
   #pragma unroll
   for(unsigned kk=0;kk<64;kk+=16){A909 a;W909 g,u;wm909::load_matrix_sync(a,xs+kk,Pitch);wm909::load_matrix_sync(g,gs+wave*16*Pitch+kk,Pitch);wm909::load_matrix_sync(u,us+wave*16*Pitch+kk,Pitch);wm909::mma_sync(gate,a,g,gate);wm909::mma_sync(up,a,u,up);}
   __syncthreads();
  }
 }
 int route=groups[group*16+(t%64&15)];if(route>=0){
  #pragma unroll
  for(unsigned j=0;j<4;++j){unsigned n=nbase+wave*16+(t%64)/16+j*4;float v=infini::ops::mxfp4_detail::fusedMoeMxfp4Activate(gate.x[j],up.x[j],size_t{2});out[size_t(route)*384+n]=__float2bfloat16_rn(v);}
 }
 __syncthreads();
 }
}
__global__ void groupHybrid909(const int32_t*ids,int32_t*g16,unsigned*n16,int32_t*g4,unsigned*n4,unsigned routes){
 __shared__ unsigned c16[1024],c4[1024];
 for(unsigned first=threadIdx.x;first<routes;first+=blockDim.x){unsigned matches=0;bool leader=true;for(unsigned p=0;p<first;++p)if(ids[p]==ids[first])leader=false;if(leader)for(unsigned p=first;p<routes;++p)matches+=ids[p]==ids[first];c16[first]=(matches+15)/16;c4[first]=(matches+3)/4;}
 __syncthreads();if(threadIdx.x==0){unsigned a=0,b=0;for(unsigned p=0;p<routes;++p){a+=c16[p];b+=c4[p];}*n16=a;*n4=b;}
 for(unsigned first=threadIdx.x;first<routes;first+=blockDim.x){if(!c16[first])continue;unsigned a=0,b=0;for(unsigned p=0;p<first;++p){a+=c16[p];b+=c4[p];}unsigned j=0;for(unsigned p=first;p<routes;++p)if(ids[p]==ids[first]){g16[(a+j/16)*16+j%16]=int32_t(p);g4[(b+j/4)*4+j%4]=int32_t(p);++j;}unsigned old=j;for(;j%16;++j)g16[(a+j/16)*16+j%16]=-1;for(j=old;j%4;++j)g4[(b+j/4)*4+j%4]=-1;}
}
struct Plan909{Context*base;int32_t*groups{};unsigned*count{};int32_t*groups4{};unsigned*count4{};};
}
extern "C" void*modern_moe_mmac_create_r909(void*raw){Plan909*t=nullptr;try{auto&c=*static_cast<Context*>(raw);DeviceGuard guard(c.device);auto&v=c.tensors;if(v[0].dtype()!=DataType::kBFloat16||!v[1].size(0)||v[1].size(0)>64||v[1].size(1)!=3584||v[4].size(1)!=768||v[2].size(1)!=16||c.activation!=2)throw std::invalid_argument("909 BF16 H3584 I384 top16 SiTU tokens1..64 only");t=new Plan909{&c};size_t routes=v[1].size(0)*16;if(cudaMalloc(&t->groups,routes*16*sizeof(int32_t))!=cudaSuccess||cudaMalloc(&t->count,sizeof(unsigned))!=cudaSuccess||cudaMalloc(&t->groups4,routes*4*sizeof(int32_t))!=cudaSuccess||cudaMalloc(&t->count4,sizeof(unsigned))!=cudaSuccess)throw std::bad_alloc();return t;}catch(const std::exception&e){last_error=e.what();if(t){if(t->groups)cudaFree(t->groups);if(t->count)cudaFree(t->count);if(t->groups4)cudaFree(t->groups4);if(t->count4)cudaFree(t->count4);delete t;}return nullptr;}}
extern "C" int modern_moe_mmac_run_r909(void*raw,int mode,int scope){try{
 auto&t=*static_cast<Plan909*>(raw);auto&c=*t.base;DeviceGuard guard(c.device);auto&v=c.tensors;if(mode<0||mode>7||(scope!=1&&scope!=2))throw std::invalid_argument("909 mode/scope invalid");if(mode==0)return modern_moe_group_r617(&c,4,scope,t.groups,t.count);
 size_t tokens=v[1].size(0),routes=tokens*16,experts=v[4].size(0);unsigned threads=256;auto*ids=reinterpret_cast<const int32_t*>(v[2].data());if(mode>=6)groupHybrid909<<<1,threads,0,c.stream>>>(ids,t.groups,t.count,t.groups4,t.count4,unsigned(routes));else group909<<<1,threads,0,c.stream>>>(ids,t.groups,t.count,unsigned(routes));
 dim3 grid(6,unsigned(routes));if(mode==1)direct909<<<grid,256,0,c.stream>>>(static_cast<B909*>(c.workspace),reinterpret_cast<const B909*>(v[1].data()),ids,reinterpret_cast<const uint8_t*>(v[4].data()),reinterpret_cast<const uint8_t*>(v[5].data()),t.groups,t.count,experts);
 else if(mode==2)shared909<64><<<grid,256,0,c.stream>>>(static_cast<B909*>(c.workspace),reinterpret_cast<const B909*>(v[1].data()),ids,reinterpret_cast<const uint8_t*>(v[4].data()),reinterpret_cast<const uint8_t*>(v[5].data()),t.groups,t.count,experts);
 else if(mode==3)shared909<66><<<grid,256,0,c.stream>>>(static_cast<B909*>(c.workspace),reinterpret_cast<const B909*>(v[1].data()),ids,reinterpret_cast<const uint8_t*>(v[4].data()),reinterpret_cast<const uint8_t*>(v[5].data()),t.groups,t.count,experts);
 else if(mode==4)shared909<68><<<grid,256,0,c.stream>>>(static_cast<B909*>(c.workspace),reinterpret_cast<const B909*>(v[1].data()),ids,reinterpret_cast<const uint8_t*>(v[4].data()),reinterpret_cast<const uint8_t*>(v[5].data()),t.groups,t.count,experts);
 else if(mode==5)shared909<72><<<grid,256,0,c.stream>>>(static_cast<B909*>(c.workspace),reinterpret_cast<const B909*>(v[1].data()),ids,reinterpret_cast<const uint8_t*>(v[4].data()),reinterpret_cast<const uint8_t*>(v[5].data()),t.groups,t.count,experts);
 if(mode==6){
  hybridTC909<68><<<grid,256,0,c.stream>>>(static_cast<B909*>(c.workspace),reinterpret_cast<const B909*>(v[1].data()),ids,reinterpret_cast<const uint8_t*>(v[4].data()),reinterpret_cast<const uint8_t*>(v[5].data()),t.groups,t.count,experts,routes);
  infini::ops::mxfp4_detail::hybridFallback909<B909,4><<<unsigned(routes*384),256,8192,c.stream>>>(static_cast<B909*>(c.workspace),reinterpret_cast<const B909*>(v[1].data()),ids,reinterpret_cast<const uint8_t*>(v[4].data()),reinterpret_cast<const uint8_t*>(v[5].data()),t.groups4,t.count4,experts,routes,t.count);
 }
 if(mode==7){
  dim3 persistent_grid(6,std::min(unsigned(routes),64u));
  persistentTC909<68><<<persistent_grid,256,0,c.stream>>>(static_cast<B909*>(c.workspace),reinterpret_cast<const B909*>(v[1].data()),ids,reinterpret_cast<const uint8_t*>(v[4].data()),reinterpret_cast<const uint8_t*>(v[5].data()),t.groups,t.count,experts,routes);
  infini::ops::mxfp4_detail::persistentFallback909<B909,4><<<320,256,8192,c.stream>>>(static_cast<B909*>(c.workspace),reinterpret_cast<const B909*>(v[1].data()),ids,reinterpret_cast<const uint8_t*>(v[4].data()),reinterpret_cast<const uint8_t*>(v[5].data()),t.groups4,t.count4,experts,routes,t.count);
 }
 if(scope==1)infini::ops::mxfp4_detail::staticW2R326U<<<unsigned(tokens*3584),256,0,c.stream>>>(reinterpret_cast<B909*>(v[0].data()),static_cast<const B909*>(c.workspace),ids,reinterpret_cast<const float*>(v[3].data()),reinterpret_cast<const uint8_t*>(v[6].data()),reinterpret_cast<const uint8_t*>(v[7].data()),tokens,size_t{16},experts,size_t{3584},size_t{384},size_t{0});
 auto e=cudaGetLastError();if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));return 0;
 }catch(const std::exception&e){last_error=e.what();return 1;}}
extern "C" int modern_moe_mmac_destroy_r909(void*raw){try{auto*t=static_cast<Plan909*>(raw);DeviceGuard guard(t->base->device);if(cudaStreamSynchronize(t->base->stream)!=cudaSuccess)throw std::runtime_error("909 drain");auto a=cudaFree(t->groups),b=cudaFree(t->count),c=cudaFree(t->groups4),d=cudaFree(t->count4);delete t;if(a!=cudaSuccess||b!=cudaSuccess||c!=cudaSuccess||d!=cudaSuccess)throw std::runtime_error("909 free");return 0;}catch(const std::exception&e){last_error=e.what();return 1;}}
extern "C" int modern_pair_domain_r909(void*a,void*b,void*stream){pair909<<<256,256,0,static_cast<cudaStream_t>(stream)>>>(static_cast<uint32_t*>(a),static_cast<uint32_t*>(b));return int(cudaGetLastError());}

extern "C" int modern_moe_mmac_groups_r909(void*raw,void*out,unsigned*count,size_t capacity){try{auto&t=*static_cast<Plan909*>(raw);DeviceGuard guard(t.base->device);size_t bytes=t.base->tensors[1].size(0)*16*16*sizeof(int32_t);if(!out||!count||capacity<bytes)throw std::invalid_argument("909groupcopy capacity");if(cudaStreamSynchronize(t.base->stream)!=cudaSuccess||cudaMemcpy(count,t.count,sizeof(unsigned),cudaMemcpyDeviceToHost)!=cudaSuccess||cudaMemcpy(out,t.groups,bytes,cudaMemcpyDeviceToHost)!=cudaSuccess)throw std::runtime_error("909groupcopy failed");return 0;}catch(const std::exception&e){last_error=e.what();return 1;}}
extern "C" int modern_moe_mmac_group_attributes_r909(void*raw,int*out){try{auto&t=*static_cast<Plan909*>(raw);DeviceGuard guard(t.base->device);cudaFuncAttributes old{},fixed{};if(!out||cudaFuncGetAttributes(&old,groupOriginal909)!=cudaSuccess||cudaFuncGetAttributes(&fixed,group909)!=cudaSuccess)throw std::runtime_error("909attributes failed");out[0]=old.maxThreadsPerBlock;out[1]=fixed.maxThreadsPerBlock;out[2]=old.numRegs;out[3]=fixed.numRegs;out[4]=int(old.sharedSizeBytes);out[5]=int(fixed.sharedSizeBytes);return 0;}catch(const std::exception&e){last_error=e.what();return 1;}}

extern "C" int modern_moe_hybrid_groups4_r909(void*raw,void*out,unsigned*count,size_t capacity){try{auto&t=*static_cast<Plan909*>(raw);DeviceGuard guard(t.base->device);size_t bytes=t.base->tensors[1].size(0)*16*4*sizeof(int32_t);if(!out||!count||capacity<bytes)throw std::invalid_argument("909g4capacity");if(cudaStreamSynchronize(t.base->stream)!=cudaSuccess||cudaMemcpy(count,t.count4,sizeof(unsigned),cudaMemcpyDeviceToHost)!=cudaSuccess||cudaMemcpy(out,t.groups4,bytes,cudaMemcpyDeviceToHost)!=cudaSuccess)throw std::runtime_error("909g4copy");return 0;}catch(const std::exception&e){last_error=e.what();return 1;}}
