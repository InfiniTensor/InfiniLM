#include "glm_moe_dsa_sparse_attention.hpp"
#include "execution/submit.hpp"
#include <cmath>
#include <algorithm>
#include <optional>
#include <set>
#include <stdexcept>
namespace infinilm::models::glm_moe_dsa {
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
namespace {
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
  size_t maximum=0;for(auto&row:selected)maximum=std::max(maximum,row.size());
  auto id_storage=alloc({maximum},rt::DataType::kInt64);
  auto key_storage=alloc({maximum,heads,256},q->dtype()),value_storage=alloc({maximum,heads,256},q->dtype());
  auto dot_storage=alloc({heads*maximum},q->dtype());
  auto df_storage=alloc({heads*maximum},rt::DataType::kFloat32),scaled_storage=alloc({heads*maximum},rt::DataType::kFloat32),logit_storage=alloc({heads*maximum},rt::DataType::kFloat32),prob_storage=alloc({heads*maximum},rt::DataType::kFloat32);
  auto rounded_storage=alloc({heads*maximum},q->dtype()),pb_storage=alloc({heads*maximum},q->dtype());
  auto prefix=[&](rt::Tensor storage,rt::Shape shape){return rt::Tensor::from_blob(storage->data(),shape,storage->dtype(),r.device());};
  for(size_t t=0;t<tokens;++t){size_t count=selected[t].size();auto ids=prefix(id_storage,{count});r.memcpyH2D(ids->data(),selected[t].data(),count*sizeof(int64_t),true);
   auto ks=prefix(key_storage,{count,heads,256}),vs=prefix(value_storage,{count,heads,256});
   ex::submit<infini::ops::IndexSelect>(r,infini::ops::IndexSelect::DefaultConfig(r.device().type()),k,ids,int64_t{0},ks);
   ex::submit<infini::ops::IndexSelect>(r,infini::ops::IndexSelect::DefaultConfig(r.device().type()),v,ids,int64_t{0},vs);
   auto dots=prefix(dot_storage,{heads,1,count});auto query=q->narrow({{0,t,1}})->view({heads,1,256});
   ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(r.device().type()),query,ks->permute({1,2,0}),std::optional<rt::Tensor>{},std::optional<float>{1},std::optional<float>{0},std::optional<int>{},std::optional<int>{},dots);
   auto df=prefix(df_storage,{heads,1,count}),scaled=prefix(scaled_storage,{heads,1,count}),rounded=prefix(rounded_storage,{heads,1,count}),logits=prefix(logit_storage,{heads,1,count}),prob=prefix(prob_storage,{heads,1,count}),pb=prefix(pb_storage,{heads,1,count});
   ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),dots,df);
   ex::submit<infini::ops::Mul>(r,infini::ops::Mul::DefaultConfig(r.device().type()),df,sf,scaled);
   ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),scaled,rounded);
   ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),rounded,logits);
   ex::submit<infini::ops::Softmax>(r,infini::ops::Softmax::DefaultConfig(r.device().type()),logits,int64_t{2},std::optional<infini::ops::DataType>{},prob);
   ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),prob,pb);
   auto out=output->narrow({{0,t,1}})->view({heads,1,256});
   ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(r.device().type()),pb,vs->permute({1,0,2}),std::optional<rt::Tensor>{},std::optional<float>{1},std::optional<float>{0},std::optional<int>{},std::optional<int>{},out);
  }r.syncStream();return output;
 }catch(...){try{r.syncStream();}catch(...){std::terminate();}throw;}
}
}
rt::Tensor sparse_attention_bounded(rt::Runtime&r,const rt::Tensor&q,const rt::Tensor&k,const rt::Tensor&v,const rt::Tensor&indices,size_t past,float scale){return sparse_impl(r,q,k,v,indices,{},past,scale);}
rt::Tensor sparse_attention_mapped_bounded(rt::Runtime&r,const rt::Tensor&q,const rt::Tensor&k,const rt::Tensor&v,const rt::Tensor&indices,const rt::Tensor&positions,size_t past,float scale){if(!positions)throw std::invalid_argument("GLM mapped key positions missing");return sparse_impl(r,q,k,v,indices,positions,past,scale);}
}
