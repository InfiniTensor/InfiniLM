#pragma once
extern "C" void*kimi_batched_create_r723(int,void*);
extern "C" int kimi_batched_run_r723(void*,int,int,int,int,int,const void*,const void*,void*);
extern "C" int kimi_batched_destroy_r723(void*);
extern "C" const char*kimi_batched_error_r723();
namespace {
using Bf727=__nv_bfloat16;
struct Tc727 {Context*base{};void*blas{};Bf727*decoded{},*packed_x{};float*gu{};int32_t*groups{};unsigned*count{};};
void cuda727(cudaError_t s){if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s));}
void blas727(int s){if(s)throw std::runtime_error(kimi_batched_error_r723());}
__global__ void groupStable727(const int32_t*ids,int32_t*groups,unsigned*count,size_t routes){
 __shared__ unsigned counts[256];size_t first=threadIdx.x;unsigned matches=0;
 if(first<routes){bool leader=true;for(size_t p=0;p<first;++p)if(ids[p]==ids[first])leader=false;if(leader)for(size_t p=first;p<routes;++p)matches+=ids[p]==ids[first];}
 counts[first]=(matches+15)/16;__syncthreads();unsigned base=0;for(size_t p=0;p<first;++p)base+=counts[p];
 if(first==255)*count=base+counts[first];
 if(matches){unsigned j=0;for(size_t p=first;p<routes;++p)if(ids[p]==ids[first]){groups[(base+j/16)*16+j%16]=int32_t(p);++j;}for(;j%16;++j)groups[(base+j/16)*16+j%16]=-1;}
}
__global__ void decode727(Bf727*out,const uint8_t*packed,const uint8_t*scales,const int32_t*ids,const int32_t*groups,size_t count,size_t experts){
 constexpr size_t K=3584,M=768,P=K/2,S=K/32;
 for(size_t p=size_t(blockIdx.x)*blockDim.x+threadIdx.x;p<count*M*P;p+=size_t(gridDim.x)*blockDim.x){size_t group=p/(M*P),row=p/P%M,col=p%P;int expert=ids[groups[group*16]];float lo=0,hi=0;if(expert>=0&&size_t(expert)<experts)infini::ops::mxfp4_detail::mxfp4DecodePair(packed[(size_t(expert)*M+row)*P+col],scales[(size_t(expert)*M+row)*S+col/16],lo,hi);out[p*2]=__float2bfloat16_rn(lo);out[p*2+1]=__float2bfloat16_rn(hi);}
}
__global__ void gather727(Bf727*out,const Bf727*x,const int32_t*groups,size_t count){
 for(size_t p=size_t(blockIdx.x)*blockDim.x+threadIdx.x;p<count*16*3584;p+=size_t(gridDim.x)*blockDim.x){int route=groups[p/3584];out[p]=route<0?__float2bfloat16_rn(0.f):x[size_t(route/16)*3584+p%3584];}
}
__global__ void activate727(Bf727*out,const float*gu,const int32_t*groups,const int32_t*ids,size_t count,size_t experts){
 for(size_t p=size_t(blockIdx.x)*blockDim.x+threadIdx.x;p<count*16*384;p+=size_t(gridDim.x)*blockDim.x){size_t row=p/384,col=p%384;int route=groups[row];if(route<0)continue;int expert=ids[route];float value=0;if(expert>=0&&size_t(expert)<experts)value=infini::ops::mxfp4_detail::fusedMoeMxfp4Activate(gu[row*768+col],gu[row*768+col+384],size_t{2});out[size_t(route)*384+col]=__float2bfloat16_rn(value);}
}
unsigned grid727(size_t n){return unsigned(std::min(size_t(65535),(n+255)/256));}
void release727(Tc727*t){if(!t)return;auto&c=*t->base;cuda727(cudaStreamSynchronize(c.stream));if(t->blas)blas727(kimi_batched_destroy_r723(t->blas));if(t->decoded)cuda727(cudaFree(t->decoded));if(t->packed_x)cuda727(cudaFree(t->packed_x));if(t->gu)cuda727(cudaFree(t->gu));if(t->groups)cuda727(cudaFree(t->groups));if(t->count)cuda727(cudaFree(t->count));delete t;}
}
extern "C" void*modern_moe_tc_create_r727(void*opaque){Tc727*t=nullptr;try{auto&c=*static_cast<Context*>(opaque);DeviceGuard guard(c.device);auto&v=c.tensors;if(v[0].dtype()!=DataType::kBFloat16||!v[1].size(0)||v[1].size(0)>16||v[1].size(1)!=3584||v[4].size(1)!=768||v[2].size(1)!=16||c.activation!=2)throw std::invalid_argument("727 supports BF16 H3584 I384 top16 SiTU decode1..16 only");size_t routes=v[1].size(0)*16;t=new Tc727;t->base=&c;t->blas=kimi_batched_create_r723(c.device,c.stream);if(!t->blas)throw std::runtime_error(kimi_batched_error_r723());cuda727(cudaMalloc(&t->decoded,routes*768*3584*sizeof(Bf727)));cuda727(cudaMalloc(&t->packed_x,routes*16*3584*sizeof(Bf727)));cuda727(cudaMalloc(&t->gu,routes*16*768*sizeof(float)));cuda727(cudaMalloc(&t->groups,routes*16*sizeof(int32_t)));cuda727(cudaMalloc(&t->count,sizeof(unsigned)));return t;}catch(const std::exception&e){last_error=e.what();if(t)try{DeviceGuard guard(t->base->device);release727(t);}catch(...){std::terminate();}return nullptr;}}
extern "C" int modern_moe_tc_run_r727(void*value,int mode,int scope,void*baseline_groups,void*baseline_count){try{
 auto&t=*static_cast<Tc727*>(value);auto&c=*t.base;DeviceGuard guard(c.device);auto&v=c.tensors;if(mode<0||mode>2||(scope!=1&&scope!=2))throw std::invalid_argument("727 mode/scope invalid");if(mode==0)return modern_moe_group_r617(&c,4,scope,baseline_groups,baseline_count);
 size_t tokens=v[1].size(0),routes=tokens*16,experts=v[4].size(0);auto*ids=reinterpret_cast<const int32_t*>(v[2].data());groupStable727<<<1,256,0,c.stream>>>(ids,t.groups,t.count,routes);unsigned count=0;cuda727(cudaMemcpyAsync(&count,t.count,sizeof(count),cudaMemcpyDeviceToHost,c.stream));cuda727(cudaStreamSynchronize(c.stream));if(!count||count>routes)throw std::runtime_error("727 group count invalid");
 // The adaptive candidate includes the grouping/counter copy even when falling back.
 if(mode==2&&size_t(count)*3>routes)return modern_moe_group_r617(&c,4,scope,baseline_groups,baseline_count);
 decode727<<<grid727(size_t(count)*768*1792),256,0,c.stream>>>(t.decoded,reinterpret_cast<const uint8_t*>(v[4].data()),reinterpret_cast<const uint8_t*>(v[5].data()),ids,t.groups,count,experts);
 gather727<<<grid727(size_t(count)*16*3584),256,0,c.stream>>>(t.packed_x,reinterpret_cast<const Bf727*>(v[1].data()),t.groups,count);
 blas727(kimi_batched_run_r723(t.blas,1,int(count),16,768,3584,t.decoded,t.packed_x,t.gu));
 activate727<<<grid727(size_t(count)*16*384),256,0,c.stream>>>(reinterpret_cast<Bf727*>(c.workspace),t.gu,t.groups,ids,count,experts);
 if(scope==1)infini::ops::mxfp4_detail::staticW2R326U<<<unsigned(tokens*3584),256,0,c.stream>>>(reinterpret_cast<Bf727*>(v[0].data()),reinterpret_cast<const Bf727*>(c.workspace),ids,reinterpret_cast<const float*>(v[3].data()),reinterpret_cast<const uint8_t*>(v[6].data()),reinterpret_cast<const uint8_t*>(v[7].data()),tokens,size_t{16},experts,size_t{3584},size_t{384},size_t{0});
 cuda727(cudaGetLastError());return 0;
 }catch(const std::exception&e){last_error=e.what();return 1;}}
extern "C" int modern_moe_tc_destroy_r727(void*value){try{auto*t=static_cast<Tc727*>(value);DeviceGuard guard(t->base->device);release727(t);return 0;}catch(const std::exception&e){last_error=e.what();return 1;}}
