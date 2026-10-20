#include "runtime/runtime.hpp"
#include "execution/submit.hpp"
#include <array>
#include <atomic>
#include <cmath>
#include <optional>
#include <stdexcept>
extern "C" int glm_scale_round_r392(const void*,void*,void*,size_t,float,void*);
namespace infinilm::models::glm_moe_dsa {
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
namespace {std::array<std::atomic<uint64_t>,8> direct_calls_r601{};}
// Internal contract: selection_map_r302(count=1) has already built sorted,
// unique causal K/V. Its mapped IDs cover the whole dense range exactly once
// after deduplication, so repeating IndexSelect is an identity copy.
rt::Tensor sparse_attention_all_selected_decode_r601(rt::Runtime&r,const rt::Tensor&q,const rt::Tensor&k,const rt::Tensor&v,float scale){
 r.assertOwnerThread();
 if(r.isGraphRecording()||r.device().type()!=rt::Device::Type::kHygon||!q||!k||!v||q->ndim()!=3||k->ndim()!=3||v->shape()!=k->shape()||q->size(0)!=1||!q->size(1)||q->size(1)>128||q->size(2)!=256||!k->size(0)||k->size(0)>2048||k->size(1)!=q->size(1)||k->size(2)!=256||q->dtype()!=rt::DataType::kBFloat16||k->dtype()!=q->dtype()||v->dtype()!=q->dtype()||q->device()!=r.device()||k->device()!=r.device()||v->device()!=r.device()||!q->is_contiguous()||!k->is_contiguous()||!v->is_contiguous()||!std::isfinite(scale)||scale<=0)throw std::invalid_argument("GLM all-selected decode requires validated contiguous causal KV");
 std::vector<rt::Tensor>held;auto alloc=[&](rt::Shape shape,rt::DataType dtype){auto x=rt::Tensor::empty(r,shape,dtype);held.push_back(x);return x;};
 try{
  size_t heads=q->size(1),count=k->size(0);auto dots=alloc({heads,1,count},q->dtype()),logits=alloc({heads,1,count},rt::DataType::kFloat32),prob=alloc({heads,1,count},rt::DataType::kFloat32),pb=alloc({heads,1,count},q->dtype()),out=alloc({1,heads,256},q->dtype());
  ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(r.device().type()),q->view({heads,1,256}),k->permute({1,2,0}),std::optional<rt::Tensor>{},std::optional<float>{1},std::optional<float>{0},std::optional<int>{},std::optional<int>{},dots);
  if(glm_scale_round_r392(dots->data(),logits->data(),nullptr,heads*count,scale,r.stream()))throw std::runtime_error("GLM direct scale/round failed");
  ex::submit<infini::ops::Softmax>(r,infini::ops::Softmax::DefaultConfig(r.device().type()),logits,int64_t{2},std::optional<infini::ops::DataType>{},prob);
  ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),prob,pb);
  ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(r.device().type()),pb,v->permute({1,0,2}),std::optional<rt::Tensor>{},std::optional<float>{1},std::optional<float>{0},std::optional<int>{},std::optional<int>{},out->view({heads,1,256}));
  r.syncStream();direct_calls_r601.at(r.device().index()).fetch_add(1,std::memory_order_relaxed);return out;
 }catch(...){try{r.syncStream();}catch(...){std::terminate();}throw;}
}
}
extern "C" uint64_t modern_glm_direct_calls_r601(int device){return device>=0&&device<8?infinilm::models::glm_moe_dsa::direct_calls_r601[size_t(device)].load():0;}
