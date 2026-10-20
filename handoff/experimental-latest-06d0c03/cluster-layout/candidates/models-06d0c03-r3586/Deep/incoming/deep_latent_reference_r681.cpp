#include "deep_selection_r356.hpp"
#include "deepseek_v4_latent_attention.hpp"
#include "execution/submit.hpp"
#include <cmath>
#include <optional>
#include <limits>
#include <set>
#include <stdexcept>
extern "C" int deep_packed_gather_r374(void*,const void*,const void*,const void*,size_t,size_t,int64_t,int64_t,int64_t,int64_t,void*);
namespace infinilm::models::deepseek_v4 {
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
rt::Tensor latent_attention_reference_r681(rt::Runtime&r,const rt::Tensor&q,const rt::Tensor&swa,const rt::Tensor&compressed,const rt::Tensor&sink,const rt::Tensor&si,const rt::Tensor&ci,const rt::Tensor&lengths){
 r.assertOwnerThread();
 if(r.isGraphRecording()||r.device().type()!=rt::Device::Type::kHygon||!q||!swa||!compressed||!sink||!si||!ci||!lengths||q->ndim()!=3||!q->size(0)||!q->size(1)||q->size(2)!=512||q->dtype()!=rt::DataType::kBFloat16||swa->ndim()!=2||compressed->ndim()!=2||swa->size(1)!=512||compressed->size(1)!=512||!swa->size(0)||!compressed->size(0)||swa->dtype()!=q->dtype()||compressed->dtype()!=q->dtype()||sink->shape()!=rt::Shape{q->size(1)}||sink->dtype()!=rt::DataType::kFloat32||si->ndim()!=2||ci->ndim()!=2||si->size(0)!=q->size(0)||ci->size(0)!=q->size(0)||!si->size(1)||!ci->size(1)||si->dtype()!=rt::DataType::kInt64||ci->dtype()!=rt::DataType::kInt64||lengths->shape()!=rt::Shape{q->size(0),2}||lengths->dtype()!=rt::DataType::kInt32)
  throw std::invalid_argument("DeepSeek latent attention requires BF16 D512 KV, F32 sink and explicit selected lengths");
 for(const auto&x:{q,swa,compressed,sink,si,ci,lengths})if(x->device()!=r.device())throw std::invalid_argument("DeepSeek latent attention owner differs");
 if(!si->is_contiguous()||!ci->is_contiguous()||!lengths->is_contiguous())throw std::invalid_argument("DeepSeek selection metadata must be contiguous");
 size_t tokens=q->size(0),heads=q->size(1),sw=si->size(1),cw=ci->size(1);std::vector<int64_t>hs(tokens*sw),hc(tokens*cw);std::vector<int32_t>hl(tokens*2);
 r.memcpyD2H(hs.data(),si->data(),hs.size()*8);r.memcpyD2H(hc.data(),ci->data(),hc.size()*8);r.memcpyD2H(hl.data(),lengths->data(),hl.size()*4);
 auto selection_r356=latent_selection_r356(hs,hc,hl,tokens,sw,cw,swa->size(0),compressed->size(0));
 auto&selected=selection_r356.selected;auto&packed_ids=selection_r356.packed_ids;auto&packed_offsets=selection_r356.packed_offsets;
 if(packed_ids.size()>std::numeric_limits<size_t>::max()/2048)throw std::invalid_argument("DeepSeek packed latent selection size overflows");
 std::vector<rt::Tensor>held;auto alloc=[&](rt::Shape shape,rt::DataType dt=rt::DataType::kFloat32){auto x=rt::Tensor::empty(r,shape,dt);held.push_back(x);return x;};
 try{auto output=alloc({tokens,heads,512},q->dtype());
  auto qf=alloc({tokens,heads,512});ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),q,qf);
  rt::Tensor packed_values;
  if(!packed_ids.empty()){
   auto ids=alloc({packed_ids.size()},rt::DataType::kInt64);r.memcpyH2D(ids->data(),packed_ids.data(),packed_ids.size()*8,true);
   packed_values=alloc({packed_ids.size(),512});
   int status=deep_packed_gather_r374(packed_values->data(),swa->data(),compressed->data(),ids->data(),packed_ids.size(),swa->size(0),swa->stride(0),swa->stride(1),compressed->stride(0),compressed->stride(1),r.stream());
   if(status)throw std::runtime_error("DeepSeek fused packed gather failed: "+std::to_string(status));
  }
  for(size_t t=0;t<tokens;++t){size_t n=selected[t].size();auto logits=alloc({heads,n+1});logits->narrow({{1,n,1}})->copy_from(r,sink->view({heads,1}));
   rt::Tensor values;
   if(n){values=packed_values->narrow({{0,packed_offsets[t],n}});
    ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(r.device().type()),qf->narrow({{0,t,1}})->view({heads,512}),values->permute({1,0}),std::optional<rt::Tensor>{},std::optional<float>{1.f/std::sqrt(512.f)},std::optional<float>{0},std::optional<int>{},std::optional<int>{},logits->narrow({{1,0,n}}));
   }
   auto probabilities=alloc({heads,n+1});ex::submit<infini::ops::Softmax>(r,infini::ops::Softmax::DefaultConfig(r.device().type()),logits,int64_t{1},std::optional<infini::ops::DataType>{},probabilities);
   auto result=alloc({heads,512});
   if(n)ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(r.device().type()),probabilities->narrow({{1,0,n}}),values,std::optional<rt::Tensor>{},std::optional<float>{1},std::optional<float>{0},std::optional<int>{},std::optional<int>{},result);
   else{std::vector<float>zero(heads*512,0);r.memcpyH2D(result->data(),zero.data(),zero.size()*4,false);}
   ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),result,output->narrow({{0,t,1}})->view({heads,512}));
  }r.syncStream();return output;
 }catch(...){try{r.syncStream();}catch(...){std::terminate();}throw;}
}
}
