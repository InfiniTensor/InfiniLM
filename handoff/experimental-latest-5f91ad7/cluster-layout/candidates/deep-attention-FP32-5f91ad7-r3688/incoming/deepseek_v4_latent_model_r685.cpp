#include "deep_selection_r356.hpp"
#include "deepseek_v4_latent_attention.hpp"
#include "execution/submit.hpp"
#include <cmath>
#include <optional>
#include <limits>
#include <set>
#include <stdexcept>
#include <array>
#include <atomic>
extern "C" int deep_packed_gather_r374(void*,const void*,const void*,const void*,size_t,size_t,int64_t,int64_t,int64_t,int64_t,void*);
extern "C" int deep_latent_validate_r681(const void*,const void*,const void*,size_t,size_t,size_t,size_t,size_t,void*,void*);
extern "C" int deep_latent_compute_r681(void*,void*,size_t,const void*,const void*,const void*,const void*,const void*,const void*,const void*,size_t,size_t,size_t,size_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,void*);
namespace infinilm::models::deepseek_v4 {
runtime::Tensor latent_attention_reference_r681(runtime::Runtime&,const runtime::Tensor&,const runtime::Tensor&,const runtime::Tensor&,const runtime::Tensor&,const runtime::Tensor&,const runtime::Tensor&,const runtime::Tensor&);

namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
namespace {std::array<std::atomic<uint64_t>,8> fused_calls{},fallback_calls{};}
rt::Tensor latent_attention(rt::Runtime&r,const rt::Tensor&q,const rt::Tensor&swa,const rt::Tensor&compressed,const rt::Tensor&sink,const rt::Tensor&si,const rt::Tensor&ci,const rt::Tensor&lengths){
 r.assertOwnerThread();
 if(r.isGraphRecording()||r.device().type()!=rt::Device::Type::kHygon||!q||!swa||!compressed||!sink||!si||!ci||!lengths||q->ndim()!=3||!q->size(0)||!q->size(1)||q->size(2)!=512||q->dtype()!=rt::DataType::kBFloat16||swa->ndim()!=2||compressed->ndim()!=2||swa->size(1)!=512||compressed->size(1)!=512||!swa->size(0)||!compressed->size(0)||swa->dtype()!=q->dtype()||compressed->dtype()!=q->dtype()||sink->shape()!=rt::Shape{q->size(1)}||sink->dtype()!=rt::DataType::kFloat32||si->ndim()!=2||ci->ndim()!=2||si->size(0)!=q->size(0)||ci->size(0)!=q->size(0)||!si->size(1)||!ci->size(1)||si->dtype()!=rt::DataType::kInt64||ci->dtype()!=rt::DataType::kInt64||lengths->shape()!=rt::Shape{q->size(0),2}||lengths->dtype()!=rt::DataType::kInt32)
  throw std::invalid_argument("DeepSeek latent attention requires BF16 D512 KV, F32 sink and explicit selected lengths");
 for(const auto&x:{q,swa,compressed,sink,si,ci,lengths})if(x->device()!=r.device())throw std::invalid_argument("DeepSeek latent attention owner differs");
 if(!si->is_contiguous()||!ci->is_contiguous()||!lengths->is_contiguous())throw std::invalid_argument("DeepSeek selection metadata must be contiguous");
 const size_t tokens=q->size(0),heads=q->size(1),sw=si->size(1),cw=ci->size(1);
 if(tokens<8||tokens>64||heads>128||sw>2048||cw>2048-sw||swa->size(0)>size_t(INT64_MAX)||compressed->size(0)>size_t(INT64_MAX)-swa->size(0)){auto result=latent_attention_reference_r681(r,q,swa,compressed,sink,si,ci,lengths);fallback_calls.at(r.device().index()).fetch_add(1);return result;}
 try{
  auto flag=rt::Tensor::empty(r,{1},rt::DataType::kInt32);int code=0;r.memcpyH2D(flag->data(),&code,sizeof(code),true);
  if(deep_latent_validate_r681(si->data(),ci->data(),lengths->data(),tokens,sw,cw,swa->size(0),compressed->size(0),flag->data(),r.stream()))throw std::runtime_error("Deep fused selection validation launch failed");
  r.memcpyD2H(&code,flag->data(),sizeof(code));
  if(code)throw std::invalid_argument("Deep fused invalid selection lengths, range or duplicates: "+std::to_string(code));
  auto out=rt::Tensor::empty(r,q->shape(),q->dtype());size_t parts=(sw+cw+64)/64;auto work=rt::Tensor::empty(r,{tokens,heads,parts,514},rt::DataType::kFloat32);
  if(deep_latent_compute_r681(out->data(),work->data(),parts,q->data(),swa->data(),compressed->data(),sink->data(),si->data(),ci->data(),lengths->data(),tokens,heads,sw,cw,q->stride(0),q->stride(1),q->stride(2),swa->stride(0),swa->stride(1),compressed->stride(0),compressed->stride(1),sink->stride(0),r.stream()))throw std::runtime_error("Deep fused attention launch failed");
  r.syncStream();fused_calls.at(r.device().index()).fetch_add(1);return out;
 }catch(...){try{r.syncStream();}catch(...){std::terminate();}throw;}
}
}
extern "C" uint64_t modern_deep_latent_fused_calls_r681(int d){return d>=0&&d<8?infinilm::models::deepseek_v4::fused_calls[size_t(d)].load():0;}
extern "C" uint64_t modern_deep_latent_fallback_calls_r681(int d){return d>=0&&d<8?infinilm::models::deepseek_v4::fallback_calls[size_t(d)].load():0;}
