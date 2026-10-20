#pragma once
#include <mma.h>
namespace {
namespace wm853=nvcuda::wmma;
using B853=__nv_bfloat16;
using A853=wm853::fragment<wm853::matrix_a,16,16,16,B853,wm853::row_major>;
using W853=wm853::fragment<wm853::matrix_b,16,16,16,B853,wm853::col_major>;
using C853=wm853::fragment<wm853::accumulator,16,16,16,float>;
__device__ __forceinline__ uint16_t fp4bits853(unsigned code,unsigned scale){
 unsigned mag=code&7,sign=(code&8)<<12;if(!mag)return uint16_t(sign);
 int exp=int(scale)+int(mag/2)-1;unsigned mant=mag>=2&&(mag&1)?64:0;
 if(exp>0&&exp<255)return uint16_t(sign|(unsigned(exp)<<7)|mant);
 return __bfloat16_as_ushort(__float2bfloat16_rn(ldexpf(mxfp4DecodeE2M1(uint8_t(code)),int(scale)-127)));
}
__global__ void pair853(uint32_t*fast,uint32_t*old){
 unsigned p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=65536)return;unsigned byte=p&255,scale=p>>8;float a,b;mxfp4DecodePair(uint8_t(byte),uint8_t(scale),a,b);
 fast[p]=uint32_t(fp4bits853(byte&15,scale))|(uint32_t(fp4bits853(byte>>4,scale))<<16);old[p]=uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(a)))|(uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(b)))<<16);
}
__global__ void groupOriginal853(const int32_t*ids,int32_t*groups,unsigned*count,unsigned routes){
 __shared__ unsigned counts[1024];unsigned first=threadIdx.x,matches=0;
 if(first<routes){bool leader=true;for(unsigned p=0;p<first;++p)if(ids[p]==ids[first])leader=false;if(leader)for(unsigned p=first;p<routes;++p)matches+=ids[p]==ids[first];}
 counts[first]=(matches+15)/16;__syncthreads();unsigned base=0;for(unsigned p=0;p<first;++p)base+=counts[p];if(first==blockDim.x-1)*count=base+counts[first];
 if(matches){unsigned j=0;for(unsigned p=first;p<routes;++p)if(ids[p]==ids[first]){groups[(base+j/16)*16+j%16]=int32_t(p);++j;}for(;j%16;++j)groups[(base+j/16)*16+j%16]=-1;}
}
__global__ void group853(const int32_t*ids,int32_t*groups,unsigned*count,unsigned routes){
 __shared__ unsigned counts[1024];
 for(unsigned first=threadIdx.x;first<routes;first+=blockDim.x){unsigned matches=0;bool leader=true;for(unsigned p=0;p<first;++p)if(ids[p]==ids[first])leader=false;if(leader)for(unsigned p=first;p<routes;++p)matches+=ids[p]==ids[first];counts[first]=(matches+15)/16;}
 __syncthreads();
 if(threadIdx.x==0){unsigned total=0;for(unsigned p=0;p<routes;++p)total+=counts[p];*count=total;}
 for(unsigned first=threadIdx.x;first<routes;first+=blockDim.x){if(!counts[first])continue;unsigned base=0;for(unsigned p=0;p<first;++p)base+=counts[p];unsigned j=0;for(unsigned p=first;p<routes;++p)if(ids[p]==ids[first]){groups[(base+j/16)*16+j%16]=int32_t(p);++j;}for(;j%16;++j)groups[(base+j/16)*16+j%16]=-1;}
}
__device__ __forceinline__ B853 weight853(const uint8_t*p,const uint8_t*s,size_t row,unsigned k){
 unsigned byte=p[row*1792+k/2],scale=s[row*112+k/32];return __ushort_as_bfloat16(fp4bits853((byte>>((k&1)*4))&15,scale));
}
// Direct fragments avoid decoded-bank allocation and host route-counter waits.
__global__ void direct853(B853*out,const B853*x,const int32_t*ids,const uint8_t*p,const uint8_t*s,const int32_t*groups,const unsigned*count,size_t experts){
 unsigned group=blockIdx.y;if(group>=*count)return;unsigned lane=threadIdx.x%64,wave=threadIdx.x/64,nbase=blockIdx.x*64+wave*16;
 int route=groups[group*16+(lane&15)],expert=ids[groups[group*16]];C853 gate,up;wm853::fill_fragment(gate,0.f);wm853::fill_fragment(up,0.f);
 if(expert>=0&&size_t(expert)<experts){
  for(unsigned k=0;k<3584;k+=16){A853 a;W853 g,u;
   #pragma unroll
   for(unsigned j=0;j<4;++j){unsigned kk=k+(lane/16)*4+j;a.x[j]=route<0?__float2bfloat16_rn(0.f):x[size_t(route/16)*3584+kk];size_t row=size_t(expert)*768+nbase+(lane&15);g.x[j]=weight853(p,s,row,kk);u.x[j]=weight853(p,s,row+384,kk);}
   wm853::mma_sync(gate,a,g,gate);wm853::mma_sync(up,a,u,up);
  }
 }
 if(route>=0){
  #pragma unroll
  for(unsigned j=0;j<4;++j){unsigned n=nbase+lane/16+j*4;float v=infini::ops::mxfp4_detail::fusedMoeMxfp4Activate(gate.x[j],up.x[j],size_t{2});out[size_t(route)*384+n]=__float2bfloat16_rn(v);}
 }
}
// Coalesced packed loads decode only one 64x64 tile at a time into LDS.
__global__ void shared853(B853*out,const B853*x,const int32_t*ids,const uint8_t*p,const uint8_t*s,const int32_t*groups,const unsigned*count,size_t experts){
 unsigned group=blockIdx.y;if(group>=*count)return;
 __shared__ __align__(16) B853 xs[16*64],gs[64*64],us[64*64];
 unsigned t=threadIdx.x,wave=t/64,nbase=blockIdx.x*64;int expert=ids[groups[group*16]];C853 gate,up;wm853::fill_fragment(gate,0.f);wm853::fill_fragment(up,0.f);
 if(expert>=0&&size_t(expert)<experts){
  for(unsigned k=0;k<3584;k+=64){
   for(unsigned q=t;q<16*64;q+=256){int route=groups[group*16+q/64];xs[q]=route<0?__float2bfloat16_rn(0.f):x[size_t(route/16)*3584+k+q%64];}
   for(unsigned q=t;q<64*32;q+=256){unsigned n=q/32,col=q%32;size_t row=size_t(expert)*768+nbase+n;unsigned kk=k/2+col;unsigned g=p[row*1792+kk],u=p[(row+384)*1792+kk],sg=s[row*112+kk/16],su=s[(row+384)*112+kk/16];reinterpret_cast<uint32_t*>(gs)[q]=uint32_t(fp4bits853(g&15,sg))|(uint32_t(fp4bits853(g>>4,sg))<<16);reinterpret_cast<uint32_t*>(us)[q]=uint32_t(fp4bits853(u&15,su))|(uint32_t(fp4bits853(u>>4,su))<<16);}
   __syncthreads();
   #pragma unroll
   for(unsigned kk=0;kk<64;kk+=16){A853 a;W853 g,u;wm853::load_matrix_sync(a,xs+kk,64);wm853::load_matrix_sync(g,gs+wave*16*64+kk,64);wm853::load_matrix_sync(u,us+wave*16*64+kk,64);wm853::mma_sync(gate,a,g,gate);wm853::mma_sync(up,a,u,up);}
   __syncthreads();
  }
 }
 int route=groups[group*16+(t%64&15)];if(route>=0){
  #pragma unroll
  for(unsigned j=0;j<4;++j){unsigned n=nbase+wave*16+(t%64)/16+j*4;float v=infini::ops::mxfp4_detail::fusedMoeMxfp4Activate(gate.x[j],up.x[j],size_t{2});out[size_t(route)*384+n]=__float2bfloat16_rn(v);}
 }
}
struct Plan853{Context*base;int32_t*groups{};unsigned*count{};};
}
extern "C" void*modern_moe_mmac_create_r853(void*raw){Plan853*t=nullptr;try{auto&c=*static_cast<Context*>(raw);DeviceGuard guard(c.device);auto&v=c.tensors;if(v[0].dtype()!=DataType::kBFloat16||!v[1].size(0)||v[1].size(0)>64||v[1].size(1)!=3584||v[4].size(1)!=768||v[2].size(1)!=16||c.activation!=2)throw std::invalid_argument("853 BF16 H3584 I384 top16 SiTU tokens1..64 only");t=new Plan853{&c};size_t routes=v[1].size(0)*16;if(cudaMalloc(&t->groups,routes*16*sizeof(int32_t))!=cudaSuccess||cudaMalloc(&t->count,sizeof(unsigned))!=cudaSuccess)throw std::bad_alloc();return t;}catch(const std::exception&e){last_error=e.what();if(t){if(t->groups)cudaFree(t->groups);if(t->count)cudaFree(t->count);delete t;}return nullptr;}}
extern "C" int modern_moe_mmac_run_r853(void*raw,int mode,int scope){try{
 auto&t=*static_cast<Plan853*>(raw);auto&c=*t.base;DeviceGuard guard(c.device);auto&v=c.tensors;if(mode<0||mode>2||(scope!=1&&scope!=2))throw std::invalid_argument("853 mode/scope invalid");if(mode==0)return modern_moe_group_r617(&c,4,scope,t.groups,t.count);
 size_t tokens=v[1].size(0),routes=tokens*16,experts=v[4].size(0);unsigned threads=256;auto*ids=reinterpret_cast<const int32_t*>(v[2].data());group853<<<1,threads,0,c.stream>>>(ids,t.groups,t.count,unsigned(routes));
 dim3 grid(6,unsigned(routes));if(mode==1)direct853<<<grid,256,0,c.stream>>>(static_cast<B853*>(c.workspace),reinterpret_cast<const B853*>(v[1].data()),ids,reinterpret_cast<const uint8_t*>(v[4].data()),reinterpret_cast<const uint8_t*>(v[5].data()),t.groups,t.count,experts);
 else shared853<<<grid,256,0,c.stream>>>(static_cast<B853*>(c.workspace),reinterpret_cast<const B853*>(v[1].data()),ids,reinterpret_cast<const uint8_t*>(v[4].data()),reinterpret_cast<const uint8_t*>(v[5].data()),t.groups,t.count,experts);
 if(scope==1)infini::ops::mxfp4_detail::staticW2R326U<<<unsigned(tokens*3584),256,0,c.stream>>>(reinterpret_cast<B853*>(v[0].data()),static_cast<const B853*>(c.workspace),ids,reinterpret_cast<const float*>(v[3].data()),reinterpret_cast<const uint8_t*>(v[6].data()),reinterpret_cast<const uint8_t*>(v[7].data()),tokens,size_t{16},experts,size_t{3584},size_t{384},size_t{0});
 auto e=cudaGetLastError();if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));return 0;
 }catch(const std::exception&e){last_error=e.what();return 1;}}
extern "C" int modern_moe_mmac_destroy_r853(void*raw){try{auto*t=static_cast<Plan853*>(raw);DeviceGuard guard(t->base->device);if(cudaStreamSynchronize(t->base->stream)!=cudaSuccess)throw std::runtime_error("853 drain");auto a=cudaFree(t->groups),b=cudaFree(t->count);delete t;if(a!=cudaSuccess||b!=cudaSuccess)throw std::runtime_error("853 free");return 0;}catch(const std::exception&e){last_error=e.what();return 1;}}
extern "C" int modern_pair_domain_r853(void*a,void*b,void*stream){pair853<<<256,256,0,static_cast<cudaStream_t>(stream)>>>(static_cast<uint32_t*>(a),static_cast<uint32_t*>(b));return int(cudaGetLastError());}

extern "C" int modern_moe_mmac_groups_r853(void*raw,void*out,unsigned*count,size_t capacity){try{auto&t=*static_cast<Plan853*>(raw);DeviceGuard guard(t.base->device);size_t bytes=t.base->tensors[1].size(0)*16*16*sizeof(int32_t);if(!out||!count||capacity<bytes)throw std::invalid_argument("853groupcopy capacity");if(cudaStreamSynchronize(t.base->stream)!=cudaSuccess||cudaMemcpy(count,t.count,sizeof(unsigned),cudaMemcpyDeviceToHost)!=cudaSuccess||cudaMemcpy(out,t.groups,bytes,cudaMemcpyDeviceToHost)!=cudaSuccess)throw std::runtime_error("853groupcopy failed");return 0;}catch(const std::exception&e){last_error=e.what();return 1;}}
extern "C" int modern_moe_mmac_group_attributes_r853(void*raw,int*out){try{auto&t=*static_cast<Plan853*>(raw);DeviceGuard guard(t.base->device);cudaFuncAttributes old{},fixed{};if(!out||cudaFuncGetAttributes(&old,groupOriginal853)!=cudaSuccess||cudaFuncGetAttributes(&fixed,group853)!=cudaSuccess)throw std::runtime_error("853attributes failed");out[0]=old.maxThreadsPerBlock;out[1]=fixed.maxThreadsPerBlock;out[2]=old.numRegs;out[3]=fixed.numRegs;out[4]=int(old.sharedSizeBytes);out[5]=int(fixed.sharedSizeBytes);return 0;}catch(const std::exception&e){last_error=e.what();return 1;}}
