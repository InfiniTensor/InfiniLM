#include "glm_moe_dsa_sparse_attention.hpp"
#include "execution/submit.hpp"
#include <cmath>
#include <algorithm>
#include <optional>
#include <set>
#include <stdexcept>
#include <cuda_runtime_api.h>
extern "C" int modern_glm_dense_causal_mask_r25(void*,size_t,size_t,size_t,size_t,void*);
namespace infinilm::models::glm_moe_dsa {
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
namespace {
thread_local int dense_mode_r48=7;thread_local bool capture_enabled_r48=false,capture_ready_r48=false;thread_local std::vector<unsigned char>capture_bytes_r48[4];
void capture_dense_r48(rt::Runtime&r,const rt::Tensor&dots,const rt::Tensor&logits,const rt::Tensor&prob,const rt::Tensor&pb){
 if(!capture_enabled_r48||capture_ready_r48)return;std::vector<rt::Tensor>values={dots,logits,prob,pb};for(size_t i=0;i<values.size();++i){capture_bytes_r48[i].resize(values[i]->nbytes());r.memcpyD2H(capture_bytes_r48[i].data(),values[i]->data(),values[i]->nbytes());}capture_ready_r48=true;
}
rt::Tensor sparse_impl(rt::Runtime&r,const rt::Tensor&q,const rt::Tensor&k,const rt::Tensor&v,const rt::Tensor&indices,const rt::Tensor&positions,size_t past,float scale){
 r.assertOwnerThread();
 if(r.isGraphRecording()||r.device().type()!=rt::Device::Type::kHygon||!q||!k||!v||!indices||q->ndim()!=3||k->ndim()!=3||v->ndim()!=3||indices->ndim()!=2||q->dtype()!=rt::DataType::kBFloat16||k->dtype()!=q->dtype()||v->dtype()!=q->dtype()||indices->dtype()!=rt::DataType::kInt64||q->device()!=r.device()||k->device()!=r.device()||v->device()!=r.device()||indices->device()!=r.device()||!q->size(0)||!q->size(1)||q->size(2)!=256||v->size(2)!=256||k->size(1)!=q->size(1)||v->size(1)!=q->size(1)||k->size(2)!=256||k->size(0)!=v->size(0)||(!positions&&(past>k->size(0)||q->size(0)>k->size(0)-past))||indices->size(0)!=q->size(0)||!indices->size(1)||indices->size(1)>2048||!std::isfinite(scale)||scale<=0)
  throw std::invalid_argument("GLM sparse attention requires explicit Hygon BF16 expanded D256/V256 geometry");
 size_t tokens=q->size(0),heads=q->size(1),width=indices->size(1);std::vector<int64_t> host(tokens*width);
 auto ci=indices->contiguous(r);r.memcpyD2H(host.data(),ci->data(),host.size()*sizeof(int64_t));
 std::vector<int64_t>logical(k->size(0));
 if(positions){if(positions->device()!=r.device()||positions->dtype()!=rt::DataType::kInt64||positions->shape()!=rt::Shape{k->size(0)}||!positions->is_contiguous())throw std::invalid_argument("GLM mapped key positions differ");r.memcpyD2H(logical.data(),positions->data(),logical.size()*8);for(auto p:logical)if(p<0)throw std::invalid_argument("GLM mapped key position is negative");}
 else for(size_t i=0;i<logical.size();++i)logical[i]=int64_t(i);
 std::vector<std::vector<int64_t>> selected(tokens);
 for(size_t t=0;t<tokens;++t){std::set<int64_t> seen;for(size_t j=0;j<width;++j){int64_t id=host[t*width+j];if(id<0||size_t(id)>=k->size(0))throw std::invalid_argument("GLM sparse selection index out of range");if(size_t(logical[size_t(id)])<=past+t)seen.insert(id);}if(seen.empty())throw std::invalid_argument("GLM sparse selection has no causal key");selected[t].assign(seen.begin(),seen.end());}
 std::vector<rt::Tensor> held;auto alloc=[&](rt::Shape shape,rt::DataType dtype){auto x=rt::Tensor::empty(r,shape,dtype);held.push_back(x);return x;};
 try{
  auto output=alloc({tokens,heads,256},q->dtype());
  auto sf=alloc({1},rt::DataType::kFloat32);sf->copy_from(r,rt::Tensor::from_blob(&scale,{1},rt::DataType::kFloat32,{rt::Device::Type::kCpu,0}));r.syncStream();
  // Keep the scaling multiplication in FP32, then round to BF16 exactly once,
  // like torch's BF16 tensor times Python scalar. Do not round the scalar first.
  // Buffers are owned until the single completion fence. Every query uses
  // contiguous prefix views and runs on this Runtime stream, so a later write
  // cannot overwrite operands still being read by an earlier query.
  // Only a complete causal prefix qualifies. Pruned/irregular selections retain
  // the original per-query implementation and duplicate-as-set semantics.
  bool dense=tokens>1 && k->size(0)==past+tokens && k->size(0)<=2048;
  for(size_t i=0;dense&&i<logical.size();++i)dense=logical[i]==int64_t(i);
  for(size_t t=0;dense&&t<tokens;++t){dense=selected[t].size()==past+t+1;for(size_t j=0;dense&&j<selected[t].size();++j)dense=selected[t][j]==int64_t(j);}
  if(dense&&dense_mode_r48!=8){
   size_t keys=k->size(0);int mode=dense_mode_r48;
   auto dots=alloc({heads,tokens,keys},q->dtype());r.setDeviceMemory(dots->data(),0,dots->nbytes());
   auto gemm=[&](const rt::Tensor&a,const rt::Tensor&b,const rt::Tensor&c){ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(r.device().type()),a,b,std::optional<rt::Tensor>{},std::optional<float>{1},std::optional<float>{0},std::optional<int>{},std::optional<int>{},c);};
   auto temp_dot=alloc({heads*keys},q->dtype());
   if(mode&1)gemm(q->permute({1,0,2}),k->permute({1,2,0}),dots);
   else for(size_t t=0;t<tokens;++t){size_t count=past+t+1;auto d=rt::Tensor::from_blob(temp_dot->data(),{heads,1,count},q->dtype(),r.device());auto query=q->narrow({{0,t,1}})->view({heads,1,256});gemm(query,k->narrow({{0,0,count}})->permute({1,2,0}),d);dots->narrow({{1,t,1},{2,0,count}})->copy_from(r,d);}
   auto df=alloc(dots->shape(),rt::DataType::kFloat32),scaled=alloc(dots->shape(),rt::DataType::kFloat32),rounded=alloc(dots->shape(),q->dtype()),logits=alloc(dots->shape(),rt::DataType::kFloat32),prob=alloc(dots->shape(),rt::DataType::kFloat32),pb=alloc(dots->shape(),q->dtype());
   ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),dots,df);
   ex::submit<infini::ops::Mul>(r,infini::ops::Mul::DefaultConfig(r.device().type()),df,sf,scaled);
   ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),scaled,rounded);
   ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),rounded,logits);
   if(modern_glm_dense_causal_mask_r25(logits->data(),heads,tokens,keys,past,r.stream()))throw std::runtime_error("GLM modes causal mask failed");
   r.setDeviceMemory(prob->data(),0,prob->nbytes());r.setDeviceMemory(pb->data(),0,pb->nbytes());
   if(mode&2){ex::submit<infini::ops::Softmax>(r,infini::ops::Softmax::DefaultConfig(r.device().type()),logits,int64_t{2},std::optional<infini::ops::DataType>{},prob);ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),prob,pb);}
   else {
    auto lbuf=alloc({heads*keys},rt::DataType::kFloat32),pbuf=alloc({heads*keys},rt::DataType::kFloat32),bbuf=alloc({heads*keys},q->dtype());
    for(size_t t=0;t<tokens;++t){size_t count=past+t+1;auto l=rt::Tensor::from_blob(lbuf->data(),{heads,1,count},rt::DataType::kFloat32,r.device()),p=rt::Tensor::from_blob(pbuf->data(),{heads,1,count},rt::DataType::kFloat32,r.device()),b=rt::Tensor::from_blob(bbuf->data(),{heads,1,count},q->dtype(),r.device());
     l->copy_from(r,logits->narrow({{1,t,1},{2,0,count}}));ex::submit<infini::ops::Softmax>(r,infini::ops::Softmax::DefaultConfig(r.device().type()),l,int64_t{2},std::optional<infini::ops::DataType>{},p);ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),p,b);prob->narrow({{1,t,1},{2,0,count}})->copy_from(r,p);pb->narrow({{1,t,1},{2,0,count}})->copy_from(r,b);
    }
   }
   if(mode&4){auto result=alloc({heads,tokens,256},q->dtype());gemm(pb,v->permute({1,0,2}),result);output->copy_from(r,result->permute({1,0,2}));}
   else {
    auto bbuf=alloc({heads*keys},q->dtype());
    for(size_t t=0;t<tokens;++t){size_t count=past+t+1;auto b=rt::Tensor::from_blob(bbuf->data(),{heads,1,count},q->dtype(),r.device());b->copy_from(r,pb->narrow({{1,t,1},{2,0,count}}));auto out=output->narrow({{0,t,1}})->view({heads,1,256});gemm(b,v->narrow({{0,0,count}})->permute({1,0,2}),out);}
   }
   r.syncStream();capture_dense_r48(r,dots,logits,prob,pb);return output;
  }

  size_t maximum=0;for(auto&row:selected)maximum=std::max(maximum,row.size());
  auto id_storage=alloc({maximum},rt::DataType::kInt64);
  auto key_storage=alloc({maximum,heads,256},q->dtype()),value_storage=alloc({maximum,heads,256},q->dtype());
  auto dot_storage=alloc({heads*maximum},q->dtype());
  auto df_storage=alloc({heads*maximum},rt::DataType::kFloat32),scaled_storage=alloc({heads*maximum},rt::DataType::kFloat32),logit_storage=alloc({heads*maximum},rt::DataType::kFloat32),prob_storage=alloc({heads*maximum},rt::DataType::kFloat32);
  auto rounded_storage=alloc({heads*maximum},q->dtype()),pb_storage=alloc({heads*maximum},q->dtype());
  auto prefix=[&](rt::Tensor storage,rt::Shape shape){return rt::Tensor::from_blob(storage->data(),shape,storage->dtype(),r.device());};
  bool capture_original=dense&&capture_enabled_r48&&!capture_ready_r48;rt::Tensor trace_d,trace_l,trace_p,trace_b;
  if(capture_original){trace_d=alloc({heads,tokens,k->size(0)},q->dtype());trace_l=alloc(trace_d->shape(),rt::DataType::kFloat32);trace_p=alloc(trace_d->shape(),rt::DataType::kFloat32);trace_b=alloc(trace_d->shape(),q->dtype());for(auto x:{trace_d,trace_l,trace_p,trace_b})r.setDeviceMemory(x->data(),0,x->nbytes());}
  for(size_t t=0;t<tokens;++t){size_t count=selected[t].size();auto ids=prefix(id_storage,{count});r.memcpyH2D(ids->data(),selected[t].data(),count*sizeof(int64_t),true);
   auto ks=prefix(key_storage,{count,heads,256}),vs=prefix(value_storage,{count,heads,256});
   ex::submit<infini::ops::IndexSelect>(r,infini::ops::IndexSelect::DefaultConfig(r.device().type()),k,ids,int64_t{0},ks);
   ex::submit<infini::ops::IndexSelect>(r,infini::ops::IndexSelect::DefaultConfig(r.device().type()),v,ids,int64_t{0},vs);
   auto dots=prefix(dot_storage,{heads,1,count});auto query=q->narrow({{0,t,1}})->view({heads,1,256});
   ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(r.device().type()),query,ks->permute({1,2,0}),std::optional<rt::Tensor>{},std::optional<float>{1},std::optional<float>{0},std::optional<int>{},std::optional<int>{},dots);
   auto df=prefix(df_storage,{heads,1,count}),scaled=prefix(scaled_storage,{heads,1,count}),rounded=prefix(rounded_storage,{heads,1,count}),logits=prefix(logit_storage,{heads,1,count}),prob=prefix(prob_storage,{heads,1,count}),pb=prefix(pb_storage,{heads,1,count});
   if(capture_original)trace_d->narrow({{1,t,1},{2,0,count}})->copy_from(r,dots);
   ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),dots,df);
   ex::submit<infini::ops::Mul>(r,infini::ops::Mul::DefaultConfig(r.device().type()),df,sf,scaled);
   ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),scaled,rounded);
   ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),rounded,logits);
   if(capture_original)trace_l->narrow({{1,t,1},{2,0,count}})->copy_from(r,logits);
   ex::submit<infini::ops::Softmax>(r,infini::ops::Softmax::DefaultConfig(r.device().type()),logits,int64_t{2},std::optional<infini::ops::DataType>{},prob);
   ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),prob,pb);
   if(capture_original){trace_p->narrow({{1,t,1},{2,0,count}})->copy_from(r,prob);trace_b->narrow({{1,t,1},{2,0,count}})->copy_from(r,pb);}
   auto out=output->narrow({{0,t,1}})->view({heads,1,256});
   ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(r.device().type()),pb,vs->permute({1,0,2}),std::optional<rt::Tensor>{},std::optional<float>{1},std::optional<float>{0},std::optional<int>{},std::optional<int>{},out);
  }r.syncStream();if(capture_original)capture_dense_r48(r,trace_d,trace_l,trace_p,trace_b);return output;
 }catch(...){try{r.syncStream();}catch(...){std::terminate();}throw;}
}
}
rt::Tensor sparse_attention_bounded(rt::Runtime&r,const rt::Tensor&q,const rt::Tensor&k,const rt::Tensor&v,const rt::Tensor&indices,size_t past,float scale){return sparse_impl(r,q,k,v,indices,{},past,scale);}
rt::Tensor sparse_attention_mapped_bounded(rt::Runtime&r,const rt::Tensor&q,const rt::Tensor&k,const rt::Tensor&v,const rt::Tensor&indices,const rt::Tensor&positions,size_t past,float scale){if(!positions)throw std::invalid_argument("GLM mapped key positions missing");return sparse_impl(r,q,k,v,indices,positions,past,scale);}
}

namespace infinilm::models::glm_moe_dsa {
extern "C" int modern_glm_dense_mode_r48_set(int mode){if(mode<0||mode>8)return 1;dense_mode_r48=mode;return 0;}
extern "C" void modern_glm_dense_capture_r48_enable(int enabled){capture_enabled_r48=enabled!=0;capture_ready_r48=false;for(auto&v:capture_bytes_r48)v.clear();}
extern "C" const void*modern_glm_dense_capture_r48_ptr(int field){return field>=0&&field<4?capture_bytes_r48[field].data():nullptr;}
extern "C" size_t modern_glm_dense_capture_r48_size(int field){return field>=0&&field<4?capture_bytes_r48[field].size():0;}
}
