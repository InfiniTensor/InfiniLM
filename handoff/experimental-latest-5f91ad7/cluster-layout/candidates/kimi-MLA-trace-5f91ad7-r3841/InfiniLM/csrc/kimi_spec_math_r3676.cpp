#include "kimi_spec_math_r3676.hpp"
#include "execution/submit.hpp"
#include "base/indexed_kimi_delta_attention.h"
#include <cuda_runtime_api.h>
#include <infini/rt/device_guard.h>
#include <set>
#include <cmath>
extern "C" int kda_rounding_r3676(int,int,int,int,int,int,int,void*,const void*,void*,const void*,const void*,const void*,const void*,const void*,const void*,const void*,const void*,const void*,const void*,void*);
namespace infinilm::models::kimi_native {
namespace ex=infinilm::execution;
static std::array<std::array<std::atomic<uint64_t>,4>,8>snapshot_counts_r3775{};
struct KdaValidationR3676:infini::ops::IndexedKimiDeltaAttention{
 template<class...A>static void check(A...a){Validate(a...);}
};
void spec_kda_roundstep_r3676(rt::Runtime&r,rt::Tensor out,rt::Tensor state,const rt::Tensor&q,const rt::Tensor&k,const rt::Tensor&v,const rt::Tensor&gate,const rt::Tensor&beta,const rt::Tensor&alog,const rt::Tensor&bias,const rt::Tensor&offsets,const rt::Tensor&read,const rt::Tensor&write,float scale,float lower,bool norm){
 r.assertOwnerThread();
 if(!spec_math_mode_r3676(r)||r.isGraphRecording())throw std::invalid_argument("3676 verifier-only eager KDA");
 KdaValidationR3676::check(ex::detail::view(out),ex::detail::view(state),ex::detail::view(q),ex::detail::view(k),ex::detail::view(v),ex::detail::view(gate),ex::detail::view(beta),ex::detail::view(alog),ex::detail::view(bias),ex::detail::view(offsets),ex::detail::view(read),ex::detail::view(write),scale,lower,norm);
 for(const auto&t:{out,state,q,k,v,gate,beta,alog,bias,offsets,read,write})if(t->device()!=r.device())throw std::invalid_argument("3676 tensor owner device mismatch");
 const size_t R=read->numel(),T=q->size(1),H=q->size(2),D=q->size(3),P=state->size(0);
 if(!R||R>16||!T||T>128||H!=12||D!=128||P<2*R+1||P>129||q->dtype()!=rt::DataType::kBFloat16||gate->dtype()!=rt::DataType::kBFloat16||offsets->dtype()!=rt::DataType::kInt32||read->dtype()!=rt::DataType::kInt32||write->dtype()!=rt::DataType::kInt32||scale!=1.0f/std::sqrt(128.0f)||lower!=-5.0f||!norm)throw std::invalid_argument("3676 bounded verified KimiTP8 BF16 geometry");
 rt::Tensor snapshot,xq,xk,xv,xg,xb;
 try{
  xq=q->contiguous(r);xk=k->contiguous(r);xv=v->contiguous(r);xg=gate->contiguous(r);xb=beta->contiguous(r);
  const char*alias_setting=std::getenv("INFINILM_KIMI_SPEC_ALIAS_R3775");bool alias=alias_setting&&std::string(alias_setting)=="1";
  if(alias_setting&&std::string(alias_setting)!="0"&&std::string(alias_setting)!="1")throw std::invalid_argument("3775 alias policy must0or1");
  if(alias)snapshot=state;else{snapshot=rt::Tensor::empty(r,state->shape(),state->dtype());snapshot->copy_from(r,state);}
  std::vector<int32_t>off(R+1),ri(R),wi(R);r.syncStream();r.memcpyD2H(off.data(),offsets->data(),off.size()*4);r.memcpyD2H(ri.data(),read->data(),R*4);r.memcpyD2H(wi.data(),write->data(),R*4);std::set<int32_t>seen;
  if(off.front()!=0||off.back()!=int32_t(T))throw std::invalid_argument("3676 packed token offsets");
  for(size_t i=0;i<R;++i)if(off[i]<0||off[i+1]<=off[i]||off[i+1]-off[i]>8||ri[i]<1||wi[i]<1||size_t(ri[i])>=P||size_t(wi[i])>=P||!seen.insert(ri[i]).second||!seen.insert(wi[i]).second)throw std::invalid_argument("3676 aliased states or invalid verification width");
  infini::rt::DeviceGuard actual_owner_guard_r3755{r.device().type(),r.device().index(),true};
  int actual=-1;if(cudaGetDevice(&actual)!=cudaSuccess||actual!=r.device().index())throw std::invalid_argument("3676 actual owning CUDA device");
  int code=kda_rounding_r3676(actual,1,int(T),int(R),int(H),int(D),int(P),out->data(),snapshot->data(),state->data(),xq->data(),xk->data(),xv->data(),xg->data(),xb->data(),alog->data(),bias->data(),offsets->data(),read->data(),write->data(),r.stream());if(code)throw std::runtime_error("3676 KDA rounded kernel launch failed");
  r.syncStream();auto&sc=snapshot_counts_r3775.at(actual);++sc[alias?0:2];sc[alias?1:3].fetch_add(state->nbytes());++math_counts_r3676.at(actual)[0];math_counts_r3676.at(actual)[1].fetch_add(T);
 }catch(...){auto first=std::current_exception();try{r.syncStream();}catch(...){std::terminate();}std::rethrow_exception(first);}
}
}
extern "C" int modern_kimi_spec_math_counts_r3676(int device,uint64_t*out,size_t count){if(device<0||device>=8||!out||count!=3)return 1;for(size_t i=0;i<3;++i)out[i]=infinilm::models::kimi_native::math_counts_r3676[device][i].load();return 0;}

namespace infinilm::models::kimi_native {
// Test-only bridge: borrowed buffers are held by the synchronous Python caller.
extern "C" int modern_kimi_guard_contract_r3755(int device,int ambient,int T,int R,int P,void**p){
 int previous=-1;cudaGetDevice(&previous);
 try{
  if(!p||device<0||device>=8||ambient<0||ambient>=8)throw std::invalid_argument("3755 contract metadata");
  auto owner=rt::Runtime::create(rt::Device(rt::Device::Type::kHygon,device));auto&r=*owner;r.enableOwnerDrainFailStop();
  {
   auto bf=rt::DataType::kBFloat16,fp=rt::DataType::kFloat32,i32=rt::DataType::kInt32;
   auto wrap=[&](int i,rt::Shape shape,rt::DataType type){return rt::Tensor::from_blob(p[i],shape,type,r.device());};
   auto out=wrap(0,{1,size_t(T),12,128},bf),state=wrap(1,{size_t(P),12,128,128},bf);
   auto q=wrap(2,{1,size_t(T),12,128},bf),k=wrap(3,{1,size_t(T),12,128},bf),v=wrap(4,{1,size_t(T),12,128},bf),g=wrap(5,{1,size_t(T),12,128},bf),b=wrap(6,{1,size_t(T),12},bf);
   auto a=wrap(7,{12},fp),bias=wrap(8,{12,128},fp),off=wrap(9,{size_t(R+1)},i32),read=wrap(10,{size_t(R)},i32),write=wrap(11,{size_t(R)},i32);
   if(cudaSetDevice(ambient)!=cudaSuccess)throw std::runtime_error("3755 select ambient");
   {SpecMathScopeR3676 scope(r,true);spec_kda_roundstep_r3676(r,out,state,q,k,v,g,b,a,bias,off,read,write,1.0f/std::sqrt(128.0f),-5.0f,true);}
   int after=-1;if(cudaGetDevice(&after)!=cudaSuccess||after!=ambient)throw std::runtime_error("3755 failed ambient restoration");r.syncStream();
  }
  r.syncStream();r.trimMemory();owner.reset();if(cudaSetDevice(previous)!=cudaSuccess)throw std::runtime_error("3755 prior device restore");return 0;
 }catch(const std::exception&e){std::fprintf(stderr,"GUARD_CONTRACT_R3755 %s\n",e.what());cudaSetDevice(previous);return 1;}
}

}

extern "C" int modern_kimi_spec_snapshot_counts_r3775(int device,uint64_t*out,size_t count){if(device<0||device>=8||!out||count!=4)return 1;for(size_t i=0;i<4;++i)out[i]=infinilm::models::kimi_native::snapshot_counts_r3775.at(device)[i].load();return 0;}
