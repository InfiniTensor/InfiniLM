#pragma once
#include "kimi_mxfp4_bits_r761.cuh"
#define mxfp4DecodePair mxfp4FastDecodePairR760
#include "kimi_mxfp4_fast_static_r760.cuh"
#include "kimi_mxfp4_fast_adaptive_r760.cuh"
#undef mxfp4DecodePair
namespace {
__global__ void pairDomain760(float*fast,float*old){unsigned p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=65536)return;unsigned packed=p&255,scale=p>>8;mxfp4FastDecodePairR760(uint8_t(packed),uint8_t(scale),fast[p*2],fast[p*2+1]);mxfp4DecodePair(uint8_t(packed),uint8_t(scale),old[p*2],old[p*2+1]);}
}
extern "C" int modern_pair_domain_r760(void*fast,void*old,void*stream){pairDomain760<<<256,256,0,static_cast<cudaStream_t>(stream)>>>(static_cast<float*>(fast),static_cast<float*>(old));return int(cudaGetLastError());}
extern "C" int modern_moe_fast_r760(void*opaque,int mode,int scope,void*groups,void*count){try{
 if(mode==0)return modern_moe_group_r617(opaque,4,scope,groups,count);
 auto&c=*static_cast<Context*>(opaque);DeviceGuard guard(c.device);auto&v=c.tensors;
 if(mode!=1||(scope!=1&&scope!=2)||!groups||!count||v[0].dtype()!=DataType::kBFloat16||!v[1].size(0)||v[1].size(0)>64||v[1].size(1)!=3584||v[4].size(1)!=768||v[2].size(1)!=16||c.activation!=2)throw std::invalid_argument("760 actual617 BF16 geometry only");
 using T=__nv_bfloat16;size_t tokens=v[1].size(0),routes=tokens*16,experts=v[4].size(0);auto*act=reinterpret_cast<T*>(c.workspace);auto*input=reinterpret_cast<const T*>(v[1].data());auto*ids=reinterpret_cast<const int32_t*>(v[2].data());auto*g=static_cast<int32_t*>(groups);auto*n=static_cast<unsigned*>(count);
 if(cudaMemsetAsync(n,0,sizeof(unsigned),c.stream)!=cudaSuccess)throw std::runtime_error("760 counter init failed");
 infini::ops::mxfp4_detail::groupRoutesR760<4><<<unsigned((routes+255)/256),256,0,c.stream>>>(ids,g,n,routes);
 infini::ops::mxfp4_detail::groupedW13R760<T,4><<<unsigned(routes*384),256,8192,c.stream>>>(act,input,ids,reinterpret_cast<const uint8_t*>(v[4].data()),reinterpret_cast<const uint8_t*>(v[5].data()),g,n,experts,routes);
 if(scope==1)infini::ops::mxfp4_detail::staticW2R760U<<<unsigned(tokens*3584),256,0,c.stream>>>(reinterpret_cast<T*>(v[0].data()),act,ids,reinterpret_cast<const float*>(v[3].data()),reinterpret_cast<const uint8_t*>(v[6].data()),reinterpret_cast<const uint8_t*>(v[7].data()),tokens,size_t{16},experts,size_t{3584},size_t{384},size_t{0});
 auto status=cudaGetLastError();if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));return 0;
 }catch(const std::exception&e){last_error=e.what();return 1;}}
