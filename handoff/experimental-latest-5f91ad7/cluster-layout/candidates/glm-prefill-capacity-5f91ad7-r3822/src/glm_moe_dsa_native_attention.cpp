#include "glm_shared_rope_r3783.hpp"
#include "glm_moe_dsa_native_attention.hpp"
#include "glm_moe_dsa_indexer_rotary.hpp"
#include "glm_moe_dsa_sparse_attention.hpp"
#include "execution/submit.hpp"
#include <set>
#include <map>
#include <cmath>
#include <algorithm>
#include <stdexcept>
namespace infinilm::models::glm_moe_dsa {
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
namespace {
struct Work{rt::Runtime&r;std::vector<rt::Tensor>held;~Work(){try{r.syncStream();}catch(...){std::terminate();}}rt::Tensor alloc(rt::Shape s,rt::DataType d){auto x=rt::Tensor::empty(r,s,d);held.push_back(x);return x;}rt::Tensor ids(const std::vector<int64_t>&v){auto x=alloc({v.size()},rt::DataType::kInt64);r.memcpyH2D(x->data(),v.data(),v.size()*8,false);return x;}};
}
NativeIndexer::NativeIndexer(rt::Runtime&r,const nlohmann::json&cfg,const engine::distributed::RankInfo&rank):owner_(&r),heads_(0){
 r.assertOwnerThread();if(r.isGraphRecording()||(rank.tp_size!=1&&rank.tp_size!=8)||rank.device!=r.device())throw std::invalid_argument("GLM native indexer owner/partition differs");
 heads_=32/size_t(rank.tp_size);try{wk_=bind_child("wk",std::make_shared<Fp8Projection>(r,128,6144,1,0));wq_=bind_child("wq_b",std::make_shared<Fp8Projection>(r,4096,2048,rank.tp_size,rank.tp_rank));norm_=bind_child("k_norm",std::make_shared<layers::LayerNorm>(r,128,1e-6,rt::DataType::kBFloat16));head_=rt::Tensor::empty_parameter(r,{heads_,6144},rt::DataType::kBFloat16);head_float_=rt::Tensor::empty(r,{heads_,6144},rt::DataType::kFloat32);bind_weight("weights_proj.weight",weights::Weight(head_,0,rank.tp_rank,rank.tp_size));rope_=shared_rope_r3783::get(r,128,64,cfg.at("max_position_embeddings"),cfg.at("rope_parameters").at("rope_theta"));r.syncStream();}catch(...){try{r.syncStream();}catch(...){std::terminate();}throw;}
}
void NativeIndexer::process_weights_after_loading(rt::Runtime&r,bool){r.assertOwnerThread();if(&r!=owner_||ready_||r.isGraphRecording())throw std::logic_error("GLM indexer postload differs");ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),head_,head_float_);r.syncStream();ready_=true;}
NativeIndexer::Features NativeIndexer::forward(rt::Runtime&r,const rt::Tensor&hidden,const rt::Tensor&q_resid,const rt::Tensor&positions)const{
 r.assertOwnerThread();if(&r!=owner_||!ready_||r.isGraphRecording())throw std::invalid_argument("GLM native indexer Runtime differs");Work work{r};size_t tokens=hidden->size(0);
 auto key=wk_->forward(r,hidden);work.held.push_back(key);auto normalized=norm_->forward(r,key);work.held.push_back(normalized);auto query=wq_->forward(r,q_resid)->view({tokens,heads_,128});work.held.push_back(query);auto rotated=indexer_rotary(r,*rope_,query,normalized->view({tokens,1,128}),positions,64);work.held.insert(work.held.end(),{rotated.first,rotated.second});
 auto input=work.alloc(hidden->shape(),rt::DataType::kFloat32),weights=work.alloc({tokens,heads_},rt::DataType::kFloat32);ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),hidden,input);ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(r.device().type()),input,head_float_->permute({1,0}),std::optional<rt::Tensor>{},std::optional<float>{1},std::optional<float>{0},std::optional<int>{},std::optional<int>{},weights);r.syncStream();return {rotated.first,rotated.second->view({tokens,128}),weights};
}
NativeAttention::NativeAttention(rt::Runtime&r,const nlohmann::json&cfg,size_t layer,const engine::distributed::RankInfo&rank,size_t rows,size_t capacity,size_t topk):owner_(&r),rows_(rows),capacity_(capacity),maximum_(cfg.at("max_position_embeddings")),topk_(topk),full_(cfg.at("indexer_types").at(layer)=="full"){
 r.assertOwnerThread();if(r.isGraphRecording()||!rows||rows>128||!capacity||capacity>maximum_||!topk||topk>2048||(cfg.at("indexer_types").at(layer)!="full"&&cfg.at("indexer_types").at(layer)!="shared"))throw std::invalid_argument("GLM persistent attention configuration differs");
 try{prepare_=std::make_shared<AttentionPrepare>(r,cfg,rank);output_=std::make_shared<AttentionOutput>(r,cfg,rank);for(auto&item:prepare_->state_dict())bind_weight(item.first,item.second);for(auto&item:output_->state_dict())bind_weight(item.first,item.second);
  if(full_){indexer_=std::make_shared<NativeIndexer>(r,cfg,rank);for(auto&item:indexer_->state_dict())bind_weight("indexer."+item.first,item.second);selector_=std::make_unique<IndexerSelectionR1581>(r,rows,capacity,32,128,topk,rank);r.setDeviceMemory(selector_->cache()->data(),0xff,selector_->cache()->nbytes());}
  latent_=rt::Tensor::empty(r,{rows,capacity,512},rt::DataType::kBFloat16);rotary_=rt::Tensor::empty(r,{rows,capacity,64},rt::DataType::kBFloat16);r.setDeviceMemory(latent_->data(),0xff,latent_->nbytes());r.setDeviceMemory(rotary_->data(),0xff,rotary_->nbytes());lengths_.assign(rows,0);r.syncStream();
 }catch(...){try{r.syncStream();}catch(...){std::terminate();}throw;}
}
void NativeAttention::process_weights_after_loading(rt::Runtime&r,bool flag){r.assertOwnerThread();if(&r!=owner_||ready_||r.isGraphRecording())throw std::invalid_argument("GLM persistent attention postload differs");prepare_->process_weights_after_loading(r,flag);output_->process_weights_after_loading(r,flag);if(indexer_)indexer_->process_weights_after_loading(r,flag);ready_=true;}
void NativeAttention::reset(rt::Runtime&r){r.assertOwnerThread();if(&r!=owner_||r.isGraphRecording())throw std::invalid_argument("GLM persistent attention reset owner differs");r.syncStream();if(selector_)selector_->reset(r);std::fill(lengths_.begin(),lengths_.end(),0);poisoned_=false;}
NativeAttention::Output NativeAttention::forward(rt::Runtime&r,const rt::Tensor&hidden,const rt::Tensor&positions,const std::vector<Request>&requests,const std::vector<rt::Tensor>&shared){
 r.assertOwnerThread();if(&r!=owner_||!ready_||poisoned_||r.isGraphRecording()||!hidden||hidden->device()!=r.device()||hidden->dtype()!=rt::DataType::kBFloat16||hidden->ndim()!=2||!hidden->is_contiguous()||!hidden->size(0)||hidden->size(0)>4096||hidden->size(1)!=6144||!positions||positions->device()!=r.device()||positions->dtype()!=rt::DataType::kInt64||!positions->is_contiguous()||positions->shape()!=rt::Shape{hidden->size(0)}||requests.empty()||(full_?!shared.empty():shared.size()!=requests.size()))throw std::invalid_argument("GLM persistent attention input/selection differs");
 size_t tokens=hidden->size(0),total=0;std::set<size_t>seen;std::vector<int64_t>pos(tokens);r.memcpyD2H(pos.data(),positions->data(),tokens*8);
 // Reject all external metadata before either indexer or latent cache mutates.
 for(size_t i=0;i<requests.size();++i){auto request=requests[i];if(request.row>=rows_||!seen.insert(request.row).second||!request.tokens||request.past!=lengths_[request.row]||request.past>capacity_||request.tokens>capacity_-request.past||request.tokens>tokens-total)throw std::invalid_argument("GLM persistent attention request row/past/count differs");for(size_t t=0;t<request.tokens;++t)if(pos[total+t]!=int64_t(request.past+t))throw std::invalid_argument("GLM persistent attention positions differ");
  if(!full_){auto ids=shared[i];if(!ids||ids->device()!=r.device()||ids->dtype()!=rt::DataType::kInt64||!ids->is_contiguous()||ids->ndim()!=2||ids->size(0)!=request.tokens||!ids->size(1)||ids->size(1)>topk_)throw std::invalid_argument("GLM shared indexer selection shape differs");std::vector<int64_t>host(ids->numel());r.memcpyD2H(host.data(),ids->data(),host.size()*8);for(size_t t=0;t<request.tokens;++t){bool causal=false;for(size_t j=0;j<ids->size(1);++j){auto id=host[t*ids->size(1)+j];if(id<0||size_t(id)>=request.past+request.tokens)throw std::invalid_argument("GLM shared selection outside valid prefix");causal|=size_t(id)<=request.past+t;}if(!causal)throw std::invalid_argument("GLM shared selection has no causal entry");}}
  total+=request.tokens;
 }
 if(total!=tokens)throw std::invalid_argument("GLM persistent attention packed count differs");
 Work work{r};try{auto prepared=prepare_->forward_latent(r,hidden,positions);work.held.insert(work.held.end(),{prepared.query,prepared.q_resid,prepared.kv_latent,prepared.k_rot});std::vector<rt::Tensor>selection;
  if(full_){auto features=indexer_->forward(r,hidden,prepared.q_resid,positions);work.held.insert(work.held.end(),{features.query,features.key,features.head_weights});auto results=selector_->forward(r,features.query,features.key,features.head_weights,requests);for(auto&result:results){work.held.insert(work.held.end(),{result.scores,result.indices});selection.push_back(result.indices);}}
  else selection=shared;
  auto attention=work.alloc({tokens,prepare_->local_heads(),256},rt::DataType::kBFloat16);size_t begin=0;
  for(size_t i=0;i<requests.size();++i){auto request=requests[i];size_t count=request.tokens,prefix=request.past+count;auto latent_row=latent_->narrow({{0,request.row,1}})->view({capacity_,512}),rotary_row=rotary_->narrow({{0,request.row,1}})->view({capacity_,64});latent_row->narrow({{0,request.past,count}})->copy_from(r,prepared.kv_latent->narrow({{0,begin,count}}));rotary_row->narrow({{0,request.past,count}})->copy_from(r,prepared.k_rot->narrow({{0,begin,count}})->view({count,64}));
   auto chosen=selection[i];std::vector<int64_t>host(chosen->numel());r.memcpyD2H(host.data(),chosen->data(),host.size()*8);std::set<int64_t>unique;for(size_t t=0;t<count;++t)for(size_t j=0;j<chosen->size(1);++j){auto id=host[t*chosen->size(1)+j];if(id<0||size_t(id)>=prefix)throw std::invalid_argument("GLM native topk outside valid prefix");if(size_t(id)<=request.past+t)unique.insert(id);}
   std::vector<int64_t>logical(unique.begin(),unique.end()),mapped(host.size());std::map<int64_t,int64_t>mapping;for(size_t j=0;j<logical.size();++j)mapping[logical[j]]=int64_t(j);
   // Future entries need no KV expansion. Replace them with a causal entry
   // already selected for the same query; duplicate entries have set semantics.
   for(size_t t=0;t<count;++t){int64_t fallback=-1;for(size_t j=0;j<chosen->size(1);++j)if(size_t(host[t*chosen->size(1)+j])<=request.past+t){fallback=mapping.at(host[t*chosen->size(1)+j]);break;}if(fallback<0)throw std::invalid_argument("GLM native topk has no causal key");for(size_t j=0;j<chosen->size(1);++j){auto id=host[t*chosen->size(1)+j];mapped[t*chosen->size(1)+j]=size_t(id)<=request.past+t?mapping.at(id):fallback;}}
   auto ids=work.ids(logical),selected_latent=work.alloc({logical.size(),512},rt::DataType::kBFloat16),selected_rotary=work.alloc({logical.size(),64},rt::DataType::kBFloat16);
   ex::submit<infini::ops::IndexSelect>(r,infini::ops::IndexSelect::DefaultConfig(r.device().type()),latent_row,ids,int64_t{0},selected_latent);ex::submit<infini::ops::IndexSelect>(r,infini::ops::IndexSelect::DefaultConfig(r.device().type()),rotary_row,ids,int64_t{0},selected_rotary);auto expanded=prepare_->expand_cached(r,selected_latent,selected_rotary->view({logical.size(),1,64}));work.held.insert(work.held.end(),{expanded.first,expanded.second});
   auto mapped_ids=work.ids(mapped)->view(chosen->shape());auto result=count==1?sparse_attention_mapped(r,prepared.query->narrow({{0,begin,count}}),expanded.first,expanded.second,mapped_ids,ids,request.past,1.f/16):sparse_attention_mapped_bounded(r,prepared.query->narrow({{0,begin,count}}),expanded.first,expanded.second,mapped_ids,ids,request.past,1.f/16);work.held.push_back(result);attention->narrow({{0,begin,count}})->copy_from(r,result);begin+=count;
  }
  auto projected=output_->forward(r,attention);work.held.insert(work.held.end(),{projected.local,projected.hidden});r.syncStream();for(auto request:requests)lengths_[request.row]=request.past+request.tokens;return {projected.hidden,selection,prepared,attention};
 }catch(...){poisoned_=true;try{r.syncStream();}catch(...){std::terminate();}throw;}
}
}
