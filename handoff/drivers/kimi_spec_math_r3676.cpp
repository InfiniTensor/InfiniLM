#include "kimi_spec_math_r3676.hpp"
#include "execution/submit.hpp"
#include "base/indexed_kimi_delta_attention.h"
#include <cuda_runtime_api.h>
#include <set>
#include <cmath>
extern "C" int kda_rounding_r3676(int,int,int,int,int,int,int,void*,const void*,void*,const void*,const void*,const void*,const void*,const void*,const void*,const void*,const void*,const void*,const void*,void*);
namespace infinilm::models::kimi_native {
namespace ex=infinilm::execution;
struct KdaValidationR3676:infini::ops::IndexedKimiDeltaAttention{
 template<class...A>static void check(A...a){Validate(a...);}
};
void spec_kda_roundstep_r3676(rt::Runtime&r,const rt::Tensor&out,const rt::Tensor&state,const rt::Tensor&q,const rt::Tensor&k,const rt::Tensor&v,const rt::Tensor&gate,const rt::Tensor&beta,const rt::Tensor&alog,const rt::Tensor&bias,const rt::Tensor&offsets,const rt::Tensor&read,const rt::Tensor&write,float scale,float lower,bool norm){
 r.assertOwnerThread();
 if(!spec_math_mode_r3676(r)||r.isGraphRecording())throw std::invalid_argument("3676 verifier-only eager KDA");
 KdaValidationR3676::check(ex::detail::view(out),ex::detail::view(state),ex::detail::view(q),ex::detail::view(k),ex::detail::view(v),ex::detail::view(gate),ex::detail::view(beta),ex::detail::view(alog),ex::detail::view(bias),ex::detail::view(offsets),ex::detail::view(read),ex::detail::view(write),scale,lower,norm);
 for(const auto&t:{out,state,q,k,v,gate,beta,alog,bias,offsets,read,write})if(t->device()!=r.device())throw std::invalid_argument("3676 tensor owner device mismatch");
 const size_t R=read->numel(),T=q->size(1),H=q->size(2),D=q->size(3),P=state->size(0);
 if(!R||R>16||!T||T>128||H!=12||D!=128||P<2*R+1||P>129||q->dtype()!=rt::DataType::kBFloat16||gate->dtype()!=rt::DataType::kBFloat16||offsets->dtype()!=rt::DataType::kInt32||read->dtype()!=rt::DataType::kInt32||write->dtype()!=rt::DataType::kInt32||scale!=1.0f/std::sqrt(128.0f)||lower!=-5.0f||!norm)throw std::invalid_argument("3676 bounded verified KimiTP8 BF16 geometry");
 rt::Tensor snapshot,xq,xk,xv,xg,xb;
 try{
  xq=q->contiguous(r);xk=k->contiguous(r);xv=v->contiguous(r);xg=gate->contiguous(r);xb=beta->contiguous(r);
  snapshot=rt::Tensor::empty(r,state->shape(),state->dtype());snapshot->copy_from(r,state);
  std::vector<int32_t>off(R+1),ri(R),wi(R);r.syncStream();r.memcpyD2H(off.data(),offsets->data(),off.size()*4);r.memcpyD2H(ri.data(),read->data(),R*4);r.memcpyD2H(wi.data(),write->data(),R*4);std::set<int32_t>seen;
  if(off.front()!=0||off.back()!=int32_t(T))throw std::invalid_argument("3676 packed token offsets");
  for(size_t i=0;i<R;++i)if(off[i]<0||off[i+1]<=off[i]||off[i+1]-off[i]>8||ri[i]<1||wi[i]<1||size_t(ri[i])>=P||size_t(wi[i])>=P||!seen.insert(ri[i]).second||!seen.insert(wi[i]).second)throw std::invalid_argument("3676 aliased states or invalid verification width");
  int actual=-1;if(cudaGetDevice(&actual)!=cudaSuccess||actual!=r.device().index())throw std::invalid_argument("3676 actual owning CUDA device");
  int code=kda_rounding_r3676(actual,1,int(T),int(R),int(H),int(D),int(P),out->data(),snapshot->data(),state->data(),xq->data(),xk->data(),xv->data(),xg->data(),xb->data(),alog->data(),bias->data(),offsets->data(),read->data(),write->data(),r.stream());if(code)throw std::runtime_error("3676 KDA rounded kernel launch failed");
  r.syncStream();++math_counts_r3676.at(actual)[0];math_counts_r3676.at(actual)[1].fetch_add(T);
 }catch(...){auto first=std::current_exception();try{r.syncStream();}catch(...){std::terminate();}std::rethrow_exception(first);}
}
}
extern "C" int modern_kimi_spec_math_counts_r3676(int device,uint64_t*out,size_t count){if(device<0||device>=8||!out||count!=3)return 1;for(size_t i=0;i<3;++i)out[i]=infinilm::models::kimi_native::math_counts_r3676[device][i].load();return 0;}
