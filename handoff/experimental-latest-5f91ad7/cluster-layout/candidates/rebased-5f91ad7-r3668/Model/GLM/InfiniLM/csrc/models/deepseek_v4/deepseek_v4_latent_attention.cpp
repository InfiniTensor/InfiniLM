#include "deepseek_v4_latent_attention.hpp"
#include "execution/submit.hpp"
#include <cmath>
#include <optional>
#include <set>
#include <stdexcept>
namespace infinilm::models::deepseek_v4 {
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
rt::Tensor latent_attention(rt::Runtime&r,const rt::Tensor&q,const rt::Tensor&swa,const rt::Tensor&compressed,const rt::Tensor&sink,const rt::Tensor&si,const rt::Tensor&ci,const rt::Tensor&lengths){
 r.assertOwnerThread();
 if(r.isGraphRecording()||r.device().type()!=rt::Device::Type::kHygon||!q||!swa||!compressed||!sink||!si||!ci||!lengths||q->ndim()!=3||!q->size(0)||!q->size(1)||q->size(2)!=512||q->dtype()!=rt::DataType::kBFloat16||swa->ndim()!=2||compressed->ndim()!=2||swa->size(1)!=512||compressed->size(1)!=512||!swa->size(0)||!compressed->size(0)||swa->dtype()!=q->dtype()||compressed->dtype()!=q->dtype()||sink->shape()!=rt::Shape{q->size(1)}||sink->dtype()!=rt::DataType::kFloat32||si->ndim()!=2||ci->ndim()!=2||si->size(0)!=q->size(0)||ci->size(0)!=q->size(0)||!si->size(1)||!ci->size(1)||si->dtype()!=rt::DataType::kInt64||ci->dtype()!=rt::DataType::kInt64||lengths->shape()!=rt::Shape{q->size(0),2}||lengths->dtype()!=rt::DataType::kInt32)
  throw std::invalid_argument("DeepSeek latent attention requires BF16 D512 KV, F32 sink and explicit selected lengths");
 for(const auto&x:{q,swa,compressed,sink,si,ci,lengths})if(x->device()!=r.device())throw std::invalid_argument("DeepSeek latent attention owner differs");
 if(!si->is_contiguous()||!ci->is_contiguous()||!lengths->is_contiguous())throw std::invalid_argument("DeepSeek selection metadata must be contiguous");
 size_t tokens=q->size(0),heads=q->size(1),sw=si->size(1),cw=ci->size(1);std::vector<int64_t>hs(tokens*sw),hc(tokens*cw);std::vector<int32_t>hl(tokens*2);
 r.memcpyD2H(hs.data(),si->data(),hs.size()*8);r.memcpyD2H(hc.data(),ci->data(),hc.size()*8);r.memcpyD2H(hl.data(),lengths->data(),hl.size()*4);
 std::vector<std::vector<int64_t>> selected(tokens);
 for(size_t t=0;t<tokens;++t){if(hl[2*t]<0||hl[2*t+1]<0||size_t(hl[2*t])>sw||size_t(hl[2*t+1])>cw)throw std::invalid_argument("DeepSeek selected lengths invalid");std::set<int64_t>seen;
  for(int bank=0;bank<2;++bank){size_t width=bank?cw:sw,limit=bank?compressed->size(0):swa->size(0);auto&host=bank?hc:hs;for(int j=0;j<hl[2*t+bank];++j){int64_t id=host[t*width+j];if(id<0||size_t(id)>=limit)throw std::invalid_argument("DeepSeek selected KV index invalid");id+=bank?int64_t(swa->size(0)):0;if(!seen.insert(id).second)throw std::invalid_argument("DeepSeek duplicate selected KV index");selected[t].push_back(id);}}
 }
 std::vector<rt::Tensor>held;auto alloc=[&](rt::Shape shape,rt::DataType dt=rt::DataType::kFloat32){auto x=rt::Tensor::empty(r,shape,dt);held.push_back(x);return x;};
 try{auto output=alloc({tokens,heads,512},q->dtype());auto bank=alloc({swa->size(0)+compressed->size(0),512},q->dtype());bank->narrow({{0,0,swa->size(0)}})->copy_from(r,swa);bank->narrow({{0,swa->size(0),compressed->size(0)}})->copy_from(r,compressed);
  auto qf=alloc({tokens,heads,512});ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),q,qf);
  for(size_t t=0;t<tokens;++t){size_t n=selected[t].size();auto logits=alloc({heads,n+1});logits->narrow({{1,n,1}})->copy_from(r,sink->view({heads,1}));
   rt::Tensor values;
   if(n){auto ids=alloc({n},rt::DataType::kInt64),raw=alloc({n,512},q->dtype());r.memcpyH2D(ids->data(),selected[t].data(),n*8,true);ex::submit<infini::ops::IndexSelect>(r,infini::ops::IndexSelect::DefaultConfig(r.device().type()),bank,ids,int64_t{0},raw);values=alloc({n,512});ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),raw,values);
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
